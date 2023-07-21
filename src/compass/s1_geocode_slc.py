#!/usr/bin/env python

'''wrapper for geocoded CSLC'''


import re
import time

import h5py
import isce3
import journal
import numpy as np
from osgeo import gdal
from s1reader.s1_reader import is_eap_correction_necessary

from compass import s1_geocode_metadata
from compass.s1_cslc_qa import QualityAssuranceCSLC
from compass.utils.browse_image import make_browse_image
from compass.utils.elevation_antenna_pattern import apply_eap_correction
from compass.utils.geo_runconfig import GeoRunConfig
from compass.utils.h5_helpers import (algorithm_metadata_to_h5group,
                                      corrections_to_h5group,
                                      flatten_metadata_to_h5group,
                                      identity_to_h5group,
                                      init_geocoded_dataset,
                                      metadata_to_h5group,
                                      DATA_PATH, METADATA_PATH, ROOT_PATH)
from compass.utils.helpers import (bursts_grouping_generator,
                                   get_time_delta_str, get_module_name)
from compass.utils.lut import cumulative_correction_luts
from compass.utils.yaml_argparse import YamlArgparse

# TEMPORARY MEASURE TODO refactor types functions to isce3 namespace
from isce3.core.types import (truncate_mantissa, to_complex32)


def _wrap_phase(phase_arr):
    # convenience function to wrap phase
    return (phase_arr + np.pi) % (2 * np.pi) - np.pi


def run(cfg: GeoRunConfig):
    '''
    Run geocode burst workflow with user-defined
    args stored in dictionary runconfig *cfg*

    Parameters
    ---------
    cfg: GeoRunConfig
        GeoRunConfig object with user runconfig options
    '''
    module_name = get_module_name(__file__)
    info_channel = journal.info(f"{module_name}.run")
    info_channel.log(f"Starting {module_name} burst")

    # Start tracking processing time
    t_start = time.perf_counter()

    # Common initializations
    image_grid_doppler = isce3.core.LUT2d()
    threshold = cfg.geo2rdr_params.threshold
    iters = cfg.geo2rdr_params.numiter
    flatten = cfg.geocoding_params.flatten

    for burst_id, bursts in bursts_grouping_generator(cfg.bursts):
        burst = bursts[0]

        date_str = burst.sensing_start.strftime("%Y%m%d")
        geo_grid = cfg.geogrids[burst_id]
        out_shape = (geo_grid.length, geo_grid.width)

        info_channel.log(f'Starting geocoding of {burst_id} for {date_str}')

        # Reinitialize the dem raster per burst to prevent raster artifacts
        # caused by modification in geocodeSlc
        dem_raster = isce3.io.Raster(cfg.dem)
        epsg = dem_raster.get_epsg()
        proj = isce3.core.make_projection(epsg)
        ellipsoid = proj.ellipsoid

        # Get output paths for current burst
        burst_id_date_key = (burst_id, date_str)
        out_paths = cfg.output_paths[burst_id_date_key]

        # Create scratch as needed
        scratch_path = out_paths.scratch_directory


        # If enabled, get range and azimuth LUTs
        t_corrections = time.perf_counter()
        if cfg.lut_params.enabled:
            rg_lut, az_lut = \
                cumulative_correction_luts(burst, dem_path=cfg.dem,
                                           tec_path=cfg.tec_file,
                                           scratch_path=scratch_path,
                                           weather_model_path=cfg.weather_model_file,
                                           rg_step=cfg.lut_params.range_spacing,
                                           az_step=cfg.lut_params.azimuth_spacing,
                                           delay_type=cfg.tropo_params.delay_type,
                                           geo2rdr_params=cfg.geo2rdr_params)
        else:
            rg_lut = isce3.core.LUT2d()
            az_lut = isce3.core.LUT2d()
        dt_corrections = get_time_delta_str(t_corrections)

        radar_grid = burst.as_isce3_radargrid()
        native_doppler = burst.doppler.lut2d
        orbit = burst.orbit

        # Get azimuth polynomial coefficients for this burst
        az_carrier_poly2d = burst.get_az_carrier_poly()

        # Extract burst boundaries
        b_bounds = np.s_[burst.first_valid_line:burst.last_valid_line,
                         burst.first_valid_sample:burst.last_valid_sample]

        # Create sliced radar grid representing valid region of the burst
        sliced_radar_grid = burst.as_isce3_radargrid()[b_bounds]

        output_hdf5 = out_paths.hdf5_path

        with h5py.File(output_hdf5, 'w') as geo_burst_h5:
            geo_burst_h5.attrs['conventions'] = "CF-1.8"
            geo_burst_h5.attrs["contact"] = np.string_("operaops@jpl.nasa.gov")
            geo_burst_h5.attrs["institution"] = np.string_("NASA JPL")
            geo_burst_h5.attrs["project_name"] = np.string_("OPERA")
            geo_burst_h5.attrs["reference_document"] = np.string_("TBD")
            geo_burst_h5.attrs["title"] = np.string_("OPERA L2_CSLC_S1 Product")

            # add type to root for GDAL recognition of datasets
            ctype = h5py.h5t.py_create(np.complex64)
            ctype.commit(geo_burst_h5['/'].id, np.string_('complex64'))

            grid_group = geo_burst_h5.require_group(DATA_PATH)
            check_eap = is_eap_correction_necessary(burst.ipf_version)

            # Initialize source/radar and destination/geo dataset into lists
            # where polarizations for radar and geo data are put into the same
            # place on their respective lists to allow data to be correctly
            # written correct place in the HDF5
            t_prep = time.perf_counter()
            rdr_data_blks = []
            geo_datasets = []
            geo_data_blks = []
            for burst in bursts:
                pol = burst.polarization

                # Load the input burst SLC
                temp_slc_path = f'{scratch_path}/{out_paths.file_name_pol}_temp.vrt'
                burst.slc_to_vrt_file(temp_slc_path)

                # Apply EAP correction if necessary
                if check_eap.phase_correction:
                    temp_slc_path_corrected = \
                        temp_slc_path.replace('_temp.vrt',
                                              '_corrected_temp.rdr')

                    apply_eap_correction(burst,
                                         temp_slc_path,
                                         temp_slc_path_corrected,
                                         check_eap)

                    # Replace the input burst if the correction is applied
                    temp_slc_path = temp_slc_path_corrected

                # Load input dataset of current polarization as array from GDAL
                # raster
                rdr_dataset = gdal.Open(temp_slc_path, gdal.GA_ReadOnly)
                rdr_data_blks.append(rdr_dataset.ReadAsArray())

                # Prepare output dataset of current polarization in HDF5
                geo_ds = init_geocoded_dataset(grid_group, pol, geo_grid,
                                               'complex64',
                                               f'{pol} geocoded CSLC image',
                                               output_cfg=cfg.output_params)
                geo_datasets.append(geo_ds)

                # Init geocoded output blocks/arrays lists to NaN
                geo_data_blks.append(
                    np.full(out_shape, np.nan + 1j * np.nan).astype(np.complex64))

            dt_prep = get_time_delta_str(t_prep)

            # Iterate over geogrid blocks that have radar data
            t_geocoding = time.perf_counter()

            # Declare names, types, and descriptions of carrier and flatten
            # outputs
            phase_names = ['azimuth_carrier_phase', 'flattening_phase']
            phase_descrs = [f'{pol} geocoded CSLC image {desc}'
                            for desc in phase_names]

            # Prepare arrays and datasets for carrier phase and flattening
            # phase
            ((carrier_phase_data_blk, carrier_phase_ds),
             (flatten_phase_data_blk, flatten_phase_ds)) = \
            [(np.full(out_shape, np.nan).astype(np.float64),
                  init_geocoded_dataset(grid_group, ds_name, geo_grid,
                                        np.float64, ds_desc,
                                        output_cfg=cfg.output_params))
                 for ds_name, ds_desc in zip(phase_names, phase_descrs)]

            # Geocode
            isce3.geocode.geocode_slc(geo_data_blocks=geo_data_blks,
                                      rdr_data_blocks=rdr_data_blks,
                                      dem_raster=dem_raster,
                                      radargrid=radar_grid,
                                      geogrid=geo_grid, orbit=orbit,
                                      native_doppler=native_doppler,
                                      image_grid_doppler=image_grid_doppler,
                                      ellipsoid=ellipsoid,
                                      threshold_geo2rdr=threshold,
                                      num_iter_geo2rdr=iters,
                                      sliced_radargrid=sliced_radar_grid,
                                      first_azimuth_line=0,
                                      first_range_sample=0,
                                      flatten=flatten, reramp=True,
                                      az_carrier=az_carrier_poly2d,
                                      rg_carrier=isce3.core.Poly2d(np.array([0])),
                                      az_time_correction=az_lut,
                                      srange_correction=rg_lut,
                                      carrier_phase_block=carrier_phase_data_blk,
                                      flatten_phase_block=flatten_phase_data_blk)

            # write geocoded data blocks to respective HDF5 datasets
            geo_datasets.extend([carrier_phase_ds,
                                 flatten_phase_ds])
            geo_data_blks.extend([_wrap_phase(carrier_phase_data_blk),
                                  _wrap_phase(flatten_phase_data_blk)])
            for cslc_dataset, cslc_data_blk in zip(geo_datasets,
                                                   geo_data_blks):
                # only convert/modify output if type not 'complex64'
                # do nothing if type is 'complex64'
                output_type = cfg.output_params.cslc_data_type
                if output_type == 'complex32':
                    cslc_data_blk = to_complex32(cslc_data_blk)
                if output_type == 'complex64_zero_mantissa':
                    # use default nonzero_mantissa_bits = 10 below
                    truncate_mantissa(cslc_data_blk)

                # write to data block HDF5
                cslc_dataset.write_direct(cslc_data_blk)

            del dem_raster # modified in geocodeSlc
            dt_geocoding = get_time_delta_str(t_geocoding)

        # Save burst corrections and metadata with new h5py File instance
        # because io.Raster things
        t_qa_meta = time.perf_counter()
        with h5py.File(output_hdf5, 'a') as geo_burst_h5:
            root_group = geo_burst_h5[ROOT_PATH]
            identity_to_h5group(root_group, burst, cfg, 'CSLC -S1',
                                cfg.product_group.product_specification_version)

            metadata_to_h5group(root_group, burst, cfg)
            algorithm_metadata_to_h5group(root_group)
            flatten_metadata_to_h5group(root_group, cfg)
            if cfg.lut_params.enabled:
                correction_group = geo_burst_h5.require_group(
                    f'{METADATA_PATH}/processing_information')
                corrections_to_h5group(correction_group, burst, cfg, rg_lut, az_lut,
                                       scratch_path,
                                       weather_model_path=cfg.weather_model_file,
                                       delay_type=cfg.tropo_params.delay_type)

            # If needed, make browse image and compute CSLC raster stats
            browse_params = cfg.browse_image_params
            if browse_params.enabled:
                make_browse_image(out_paths.browse_path, output_hdf5,
                                  bursts, browse_params.complex_to_real,
                                  browse_params.percent_low,
                                  browse_params.percent_high,
                                  browse_params.gamma, browse_params.equalize)

            # If needed, perform QA and write results to JSON
            if cfg.quality_assurance_params.perform_qa:
                cslc_qa = QualityAssuranceCSLC()
                if cfg.lut_params.enabled:
                    # apply tropo corrections if weather file provided
                    apply_tropo_corrections = cfg.weather_model_file is not None
                    cslc_qa.compute_correction_stats(
                        geo_burst_h5, apply_tropo_corrections,
                        cfg.tropo_params.delay_type)
                cslc_qa.compute_CSLC_raster_stats(geo_burst_h5, bursts)
                cslc_qa.populate_rfi_dict(geo_burst_h5, bursts)
                cslc_qa.valid_pixel_percentages(geo_burst_h5)
                cslc_qa.set_orbit_type(cfg, geo_burst_h5)
                if cfg.quality_assurance_params.output_to_json:
                    cslc_qa.write_qa_dicts_to_json(out_paths.stats_json_path)

            if burst.burst_calibration is not None:
                # Geocode the calibration parameters and write them into HDF5
                s1_geocode_metadata.geocode_calibration_luts(geo_burst_h5,
                                                             burst,
                                                             cfg)

            if burst.burst_noise is not None:
                # Geocode the calibration parameters and write them into HDF5
                s1_geocode_metadata.geocode_noise_luts(geo_burst_h5,
                                                       burst,
                                                       cfg)
        dt_qa_meta = get_time_delta_str(t_qa_meta)

    dt = get_time_delta_str(t_start)
    info_channel.log(f"{module_name} corrections computation time {dt_corrections} (hr:min:sec)")
    info_channel.log(f"{module_name} geocode prep time {dt_prep} (hr:min:sec)")
    info_channel.log(f"{module_name} geocoding time {dt_geocoding} (hr:min:sec)")
    info_channel.log(f"{module_name} QA meta processing time {dt_qa_meta} (hr:min:sec)")
    info_channel.log(f"{module_name} burst successfully ran in {dt} (hr:min:sec)")


if __name__ == "__main__":
    '''Run geocode cslc workflow from command line'''
    # load arguments from command line
    parser = YamlArgparse()

    # Get a runconfig dict from command line arguments
    cfg = GeoRunConfig.load_from_yaml(parser.run_config_path,
                                      workflow_name='s1_cslc_geo')

    # Run geocode burst workflow
    run(cfg)

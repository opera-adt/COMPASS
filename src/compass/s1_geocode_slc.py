#!/usr/bin/env python

'''wrapper for geocoded CSLC'''

from datetime import timedelta
import time

import h5py
import isce3
from isce3.block_processing.rdr_geo_block_generator import block_generator
import journal
import numpy as np
from osgeo import gdal
from s1reader.s1_reader import is_eap_correction_necessary

from compass import s1_rdr2geo
from compass import s1_geocode_metadata
from compass.s1_cslc_qa import QualityAssuranceCSLC
from compass.utils.browse_image import make_browse_image
from compass.utils.elevation_antenna_pattern import apply_eap_correction
from compass.utils.geo_runconfig import GeoRunConfig
from compass.utils.h5_helpers import (corrections_to_h5group,
                                      identity_to_h5group,
                                      init_geocoded_dataset,
                                      metadata_to_h5group)
from compass.utils.helpers import bursts_grouping_generator, get_module_name
from compass.utils.lut import cumulative_correction_luts
from compass.utils.yaml_argparse import YamlArgparse


def _slice_to_start_and_size(grid_slice):
    size = grid_slice.stop - grid_slice.start
    return grid_slice.start, size


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
    t_start = time.time()

    # Common initializations
    image_grid_doppler = isce3.core.LUT2d()
    threshold = cfg.geo2rdr_params.threshold
    iters = cfg.geo2rdr_params.numiter
    lines_per_block = cfg.geocoding_params.lines_per_block
    cols_per_block = cfg.geocoding_params.columns_per_block
    flatten = cfg.geocoding_params.flatten
    geogrid_expansion_threshold = 100

    for burst_id, bursts in bursts_grouping_generator(cfg.bursts):
        burst = bursts[0]

        # Reinitialize the dem raster per burst to prevent raster artifacts
        # caused by modification in geocodeSlc
        dem_raster = isce3.io.Raster(cfg.dem)
        epsg = dem_raster.get_epsg()
        proj = isce3.core.make_projection(epsg)
        ellipsoid = proj.ellipsoid

        date_str = burst.sensing_start.strftime("%Y%m%d")
        geo_grid = cfg.geogrids[burst_id]

        # Get output paths for current burst
        burst_id_date_key = (burst_id, date_str)
        out_paths = cfg.output_paths[burst_id_date_key]

        # Create scratch as needed
        scratch_path = out_paths.scratch_directory


        # If enabled, get range and azimuth LUTs
        if cfg.lut_params.enabled:
            rg_lut, az_lut = \
                cumulative_correction_luts(burst, dem_path=cfg.dem,
                                           tec_path=cfg.tec_file,
                                           scratch_path=scratch_path,
                                           weather_model_path=cfg.weather_model_file,
                                           rg_step=cfg.lut_params.range_spacing,
                                           az_step=cfg.lut_params.azimuth_spacing,
                                           delay_type=cfg.tropo_params.delay_type)
        else:
            rg_lut = isce3.core.LUT2d()
            az_lut = isce3.core.LUT2d()

        radar_grid = burst.as_isce3_radargrid()
        native_doppler = burst.doppler.lut2d
        orbit = burst.orbit

        # Get azimuth polynomial coefficients for this burst
        az_carrier_poly2d = burst.get_az_carrier_poly()

        # Generate required static layers
        if cfg.rdr2geo_params.enabled:
            s1_rdr2geo.run(cfg, save_in_scratch=True)
            if cfg.rdr2geo_params.geocode_metadata_layers:
                s1_geocode_metadata.run(cfg, burst, fetch_from_scratch=True)

        # Extract burst boundaries
        b_bounds = np.s_[burst.first_valid_line:burst.last_valid_line,
                         burst.first_valid_sample:burst.last_valid_sample]

        # Create sliced radar grid representing valid region of the burst
        sliced_radar_grid = burst.as_isce3_radargrid()[b_bounds]

        output_hdf5 = out_paths.hdf5_path
        root_path = '/science/SENTINEL1'
        with h5py.File(output_hdf5, 'w') as geo_burst_h5:
            geo_burst_h5.attrs['Conventions'] = "CF-1.8"
            geo_burst_h5.attrs["contact"] = np.string_("operaops@jpl.nasa.gov")
            geo_burst_h5.attrs["institution"] = np.string_("NASA JPL")
            geo_burst_h5.attrs["mission_name"] = np.string_("OPERA")
            geo_burst_h5.attrs["reference_document"] = np.string_("TBD")
            geo_burst_h5.attrs["title"] = np.string_("OPERA L2_CSLC_S1 Product")

            # add type to root for GDAL recognition of datasets
            ctype = h5py.h5t.py_create(np.complex64)
            ctype.commit(geo_burst_h5['/'].id, np.string_('complex64'))

            grid_path = f'{root_path}/CSLC/grids'
            grid_group = geo_burst_h5.require_group(grid_path)
            check_eap = is_eap_correction_necessary(burst.ipf_version)

            # initialize source/rdr and destination/geo datasets
            # also apply EAP per raster if needed
            rdr_datasets = []
            geo_datasets = []
            for burst in bursts:
                pol = burst.polarization

                # Load the input burst SLC
                temp_slc_path = f'{scratch_path}/{out_paths.file_name_pol}_temp.vrt'
                burst.slc_to_vrt_file(temp_slc_path)

                # Apply EAP correction if necessary
                if check_eap.phase_correction:
                    temp_slc_path_corrected = temp_slc_path.replace('_temp.vrt',
                                                                    '_corrected_temp.rdr')
                    apply_eap_correction(burst,
                                         temp_slc_path,
                                         temp_slc_path_corrected,
                                         check_eap)

                    # Replace the input burst if the correction is applied
                    temp_slc_path = temp_slc_path_corrected

                # prepare input dataset as GDAL raster
                rdr_dataset = gdal.Open(temp_slc_path, gdal.GA_ReadOnly)
                print(rdr_dataset.RasterXSize, rdr_dataset.RasterYSize)
                rdr_datasets.append(rdr_dataset)

                # prepare output dataset in HDF5
                geo_ds = init_geocoded_dataset(grid_group, pol, geo_grid,
                                               'complex64',
                                               f'{pol} geocoded CSLC image')
                geo_datasets.append(geo_ds)

            # block proc things
            max_start_col = -1
            min_start_col = 100000000
            max_start_line = -1
            min_start_line = 100000000
            max_width = -1
            min_width = 100000000
            max_length = -1
            min_length = 100000000
            for (rdr_blk_slice, geo_blk_slice, geo_blk_shape,
                 blk_geo_grid) in block_generator(geo_grid, radar_grid, orbit,
                                                  dem_raster, lines_per_block,
                                                  cols_per_block,
                                                  geogrid_expansion_threshold):
                # extract start and size from slice
                start_col, width = _slice_to_start_and_size(rdr_blk_slice[0])
                start_line, length = _slice_to_start_and_size(rdr_blk_slice[1])
                #print(start_col, width, start_line, width)
                max_start_col = max(max_start_col, start_col)
                min_start_col = min(min_start_col, start_col)
                max_start_line = max(max_start_line, start_line)
                min_start_line = min(min_start_line, start_line)
                max_width = max(max_width, width)
                min_width = min(min_width, width)
                max_length = max(max_length, length)
                min_length = min(min_length, length)

                # unpack block parameters
                az_first = rdr_blk_slice[0].start
                rg_first = rdr_blk_slice[1].start

                # Init input and output blocks/arrays lists
                geo_data_blks = []
                rdr_data_blks = []

                # Build input and output arrays to be passed to geocode_slc
                for rdr_dataset in rdr_datasets:
                    rdr_data_blks.append(
                        #rdr_dataset.ReadAsArray(start_col, start_line,
                        #                        length, width))
                        rdr_dataset.ReadAsArray(start_line, start_col,
                                                length, width))

                    geo_data_blks.append(np.zeros(geo_blk_shape,
                                                  dtype=np.complex64))

                # Geocode
                isce3.geocode.geocode_slc(geo_data_blks, rdr_data_blks,
                                          dem_raster, radar_grid,
                                          sliced_radar_grid, blk_geo_grid,
                                          orbit, native_doppler,
                                          image_grid_doppler, ellipsoid,
                                          threshold, iters, start_col,
                                          start_line)#, flatten,
                '''
                                          azimuth_carrier=az_carrier_poly2d,
                                          az_time_correction=az_lut,
                                          srange_correction=rg_lut)
                '''

                # write geocoded blocks to respective HDF5 datasets
                for geo_dataset, geo_data_blk in zip(geo_datasets,
                                                     geo_data_blks):
                    # write to GSLC block HDF5
                    geo_dataset.write_direct(geo_data_blk,
                                             dest_sel=geo_blk_slice)

            print('min/max start col ', min_start_col, max_start_col)
            print('min/max width     ', min_width, max_width)
            print('min/max start line', min_start_line, max_start_line)
            print('min/max length    ', min_length, max_length)
            # Set geo transformation
            '''
            geotransform = [geo_grid.start_x, geo_grid.spacing_x, 0,
                            geo_grid.start_y, 0, geo_grid.spacing_y]
            geo_burst_raster.set_geotransform(geotransform)
            geo_burst_raster.set_epsg(epsg)
            del geo_burst_raster
            '''
            del dem_raster # modified in geocodeSlc

        # Save burst corrections and metadata with new h5py File instance
        # because io.Raster things
        with h5py.File(output_hdf5, 'a') as geo_burst_h5:
            root_group = geo_burst_h5[root_path]
            identity_to_h5group(root_group, burst, cfg)

            cslc_group = geo_burst_h5.require_group(f'{root_path}/CSLC')
            metadata_to_h5group(cslc_group, burst, cfg)
            if cfg.lut_params.enabled:
                corrections_to_h5group(cslc_group, burst, cfg, rg_lut, az_lut,
                                       scratch_path,
                                       weather_model_path=cfg.weather_model_file,
                                       delay_type=cfg.tropo_params.delay_type)

            # If needed, make browse image and compute CSLC raster stats
            browse_params = cfg.browse_image_params
            if browse_params.enabled:
                make_browse_image(out_paths.browse_path, output_hdf5,
                                  cfg.bursts, browse_params.complex_to_real,
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
                cslc_qa.populate_rfi_dict(geo_burst_h5)
                cslc_qa.valid_pixel_percentages(geo_burst_h5)
                cslc_qa.set_orbit_type(cfg, geo_burst_h5)
                if cslc_qa.output_to_json:
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

    dt = str(timedelta(seconds=time.time() - t_start)).split(".")[0]
    info_channel.log(f"{module_name} burst successfully ran in {dt} (hr:min:sec)")


if __name__ == "__main__":
    '''Run geocode cslc workflow from command line'''
    # load arguments from command line
    parser = YamlArgparse()

    # Get a runconfig dict from command line argumens
    cfg = GeoRunConfig.load_from_yaml(parser.run_config_path,
                                      workflow_name='s1_cslc_geo')

    # Run geocode burst workflow
    run(cfg)

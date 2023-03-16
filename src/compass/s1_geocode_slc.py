#!/usr/bin/env python

'''wrapper for geocoded CSLC'''

from datetime import timedelta
import itertools
import time

import h5py
import isce3
import journal
import numpy as np
from s1reader.s1_reader import is_eap_correction_necessary

from compass import s1_rdr2geo
from compass import s1_geocode_metadata
from compass.utils.elevation_antenna_pattern import apply_eap_correction
from compass.utils.geo_runconfig import GeoRunConfig
from compass.utils.h5_helpers import (corrections_to_h5group,
                                      identity_to_h5group,
                                      init_geocoded_dataset,
                                      metadata_to_h5group)
from compass.utils.helpers import get_module_name
from compass.utils.lut import cumulative_correction_luts
from compass.utils.yaml_argparse import YamlArgparse


def _bursts_grouping_generator(bursts):
    # Dict to group bursts with the same burst ID but different polarizations
    # key: burst ID, value: list[S1BurstSlc]
    grouped_bursts = itertools.groupby(bursts, key=lambda b: str(b.burst_id))

    for k, v in grouped_bursts:
        yield k, list(v)


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
    blocksize = cfg.geo2rdr_params.lines_per_block
    flatten = cfg.geocoding_params.flatten

    for burst_id, bursts in _bursts_grouping_generator(cfg.bursts):
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

            rg_lut, az_lut = cumulative_correction_luts(burst,
                                                        dem_path=cfg.dem,
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

        # Generate required metadata layers
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
            for b in bursts:
                pol = b.polarization

                # Load the input burst SLC
                temp_slc_path = f'{scratch_path}/{out_paths.file_name_pol}_temp.vrt'
                burst.slc_to_vrt_file(temp_slc_path)

                # Apply EAP correction if necessary
                if check_eap.phase_correction:
                    temp_slc_path_corrected = temp_slc_path.replace('_temp.vrt',
                                                                    '_corrected_temp.rdr')
                    apply_eap_correction(b,
                                         temp_slc_path,
                                         temp_slc_path_corrected,
                                         check_eap)

                    # Replace the input burst if the correction is applied
                    temp_slc_path = temp_slc_path_corrected


                rdr_burst_raster = isce3.io.Raster(temp_slc_path)

                init_geocoded_dataset(grid_group, pol, geo_grid, 'complex64',
                                      f'{pol} geocoded CSLC image')

                # access the HDF5 dataset for a given frequency and polarization
                dataset_path = f'{grid_path}/{pol}'
                gslc_dataset = geo_burst_h5[dataset_path]

                # Construct the output raster directly from HDF5 dataset
                geo_burst_raster = isce3.io.Raster(f"IH5:::ID={gslc_dataset.id.id}".encode("utf-8"),
                                                   update=True)

                # Geocode
                isce3.geocode.geocode_slc(geo_burst_raster, rdr_burst_raster,
                                          dem_raster,
                                          radar_grid, sliced_radar_grid,
                                          geo_grid, orbit,
                                          native_doppler,
                                          image_grid_doppler, ellipsoid, threshold,
                                          iters, blocksize, flatten,
                                          azimuth_carrier=az_carrier_poly2d,
                                          az_time_correction=az_lut,
                                          srange_correction=rg_lut)

            # Set geo transformation
            geotransform = [geo_grid.start_x, geo_grid.spacing_x, 0,
                            geo_grid.start_y, 0, geo_grid.spacing_y]
            geo_burst_raster.set_geotransform(geotransform)
            geo_burst_raster.set_epsg(epsg)
            del geo_burst_raster
            del dem_raster # modified in geocodeSlc

        # Save burst corrections and metadata with new h5py File instance
        # because io.Raster things
        with h5py.File(output_hdf5, 'a') as geo_burst_h5:
            root_group = geo_burst_h5[root_path]
            identity_to_h5group(root_group, burst)

            cslc_group = geo_burst_h5.require_group(f'{root_path}/CSLC')
            metadata_to_h5group(cslc_group, burst, cfg)
            corrections_to_h5group(cslc_group, burst, cfg, rg_lut, az_lut, scratch_path,
                                   weather_model_path=cfg.weather_model_file,
                                   delay_type=cfg.tropo_params.delay_type)

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

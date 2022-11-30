#!/usr/bin/env python

'''wrapper for geocoded SLC'''

from datetime import timedelta
import time

import h5py
import isce3
import journal
from nisar.workflows.h5_prep import set_get_geo_info
import numpy as np

from compass import s1_rdr2geo
from compass import s1_geocode_metadata
from compass.utils.geo_runconfig import GeoRunConfig
from compass.utils.h5_helpers import (init_geocoded_dataset,
                                      geo_burst_metadata_to_hdf5)
from compass.utils.helpers import get_module_name
from compass.utils.lut import compute_geocoding_correction_luts
from compass.utils.range_split_spectrum import range_split_spectrum
from compass.utils.yaml_argparse import YamlArgparse


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

    for burst in cfg.bursts:
        # Reinitialize the dem raster per burst to prevent raster artifacts
        # caused by modification in geocodeSlc
        dem_raster = isce3.io.Raster(cfg.dem)
        epsg = dem_raster.get_epsg()
        proj = isce3.core.make_projection(epsg)
        ellipsoid = proj.ellipsoid

        date_str = burst.sensing_start.strftime("%Y%m%d")
        burst_id = str(burst.burst_id)
        pol = burst.polarization
        geo_grid = cfg.geogrids[burst_id]

        # Get range and azimuth LUTs
        rg_lut, az_lut = compute_geocoding_correction_luts(burst,
                                                           rg_step=cfg.lut_params.range_spacing,
                                                           az_step=cfg.lut_params.azimuth_spacing)

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

        # Get output paths for current burst
        burst_id_date_key = (burst_id, date_str)
        out_paths = cfg.output_paths[burst_id_date_key]

        # Create scratch as needed
        scratch_path = out_paths.scratch_directory

        # Split the range bandwidth of the burst, if required
        if cfg.split_spectrum_params.enabled:
            rdr_burst_raster = range_split_spectrum(burst,
                                                    cfg.split_spectrum_params,
                                                    scratch_path)
        else:
            id_date_pol = f"{burst_id}_{date_str}_{pol}"
            temp_slc_path = f'{scratch_path}/{id_date_pol}_temp.vrt'
            burst.slc_to_vrt_file(temp_slc_path)
            rdr_burst_raster = isce3.io.Raster(temp_slc_path)

        # Extract burst boundaries
        b_bounds = np.s_[burst.first_valid_line:burst.last_valid_line,
                         burst.first_valid_sample:burst.last_valid_sample]

        # Create sliced radar grid representing valid region of the burst
        sliced_radar_grid = burst.as_isce3_radargrid()[b_bounds]

        output_hdf5 = out_paths.hdf5_path
        with h5py.File(output_hdf5, 'w') as geo_burst_h5:
            backscatter_group = geo_burst_h5.require_group('complex_backscatter')
            init_geocoded_dataset(backscatter_group, pol, geo_grid,
                                  'complex64', f'{pol} geocoded SLC image')

            # access the HDF5 dataset for a given frequency and polarization
            dataset_path = f'/complex_backscatter/{pol}'
            gslc_dataset = geo_burst_h5[dataset_path]

            # Construct the output raster directly from HDF5 dataset
            geo_burst_raster = isce3.io.Raster(f"IH5:::ID={gslc_dataset.id.id}".encode("utf-8"), update=True)

            # Geocode
            isce3.geocode.geocode_slc(geo_burst_raster, rdr_burst_raster,
                                      dem_raster,
                                      radar_grid, sliced_radar_grid,
                                      geo_grid, orbit,
                                      native_doppler,
                                      image_grid_doppler, ellipsoid, threshold,
                                      iters, blocksize, flatten,
                                      azimuth_carrier=az_carrier_poly2d)

            # Set geo transformation
            geotransform = [geo_grid.start_x, geo_grid.spacing_x, 0,
                            geo_grid.start_y, 0, geo_grid.spacing_y]
            geo_burst_raster.set_geotransform(geotransform)
            geo_burst_raster.set_epsg(epsg)
            del geo_burst_raster
            del dem_raster # modified in geocodeSlc

        # Save burst metadata with new h5py File instance because io.Raster things
        with h5py.File(output_hdf5, 'a') as geo_burst_h5:
            geo_burst_metadata_to_hdf5(geo_burst_h5, burst, geo_grid)
            geo_burst_h5['metadata/runconfig'] = cfg.yaml_string


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

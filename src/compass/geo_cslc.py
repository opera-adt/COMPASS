import os
import time

import isce3
import journal
import numpy as np
from osgeo import gdal

from compass.utils.geo_runconfig import GeoRunConfig
from compass.utils.range_split_spectrum import range_split_spectrum
from compass.utils.yaml_argparse import YamlArgparse


def run(cfg):
    '''
    Run geocode burst workflow with user-defined
    args stored in dictionary runconfig *cfg*

    Parameters
    ---------
    cfg: dict
        Dictionary with user runconfig options
    '''
    info_channel = journal.info("geo_cslc.run")

    # Start tracking processing time
    t_start = time.time()
    info_channel.log("Starting geocode burst")

    # Common initializations
    dem_raster = isce3.io.Raster(cfg.dem)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid
    image_grid_doppler = isce3.core.LUT2d()
    threshold = cfg.geo2rdr_params.threshold
    iters = cfg.geo2rdr_params.numiter
    blocksize = cfg.geo2rdr_params.lines_per_block
    dem_margin = cfg.geocoding_params.dem_margin
    flatten = cfg.geocoding_params.flatten

    for burst in cfg.bursts:
        date_str = burst.sensing_start.strftime("%Y%m%d")
        burst_id = burst.burst_id
        pol = burst.polarization
        geo_grid = cfg.geogrids[burst_id]

        os.makedirs(cfg.output_dir, exist_ok=True)

        scratch_path = f'{cfg.scratch_path}/{burst_id}/{date_str}'
        os.makedirs(scratch_path, exist_ok=True)

        radar_grid = burst.as_isce3_radargrid()
        native_doppler = burst.doppler.lut2d
        orbit = burst.orbit

        # Get azimuth polynomial coefficients for this burst
        az_carrier_poly2d = burst.get_az_carrier_poly()

        # Split the range bandwidth of the burst, if required
        if cfg.split_spectrum_params.enabled:
            rdr_burst_raster = range_split_spectrum(burst,
                                                    cfg.split_spectrum_params,
                                                    scratch_path)
        else:
            temp_slc_path = f'{scratch_path}/{burst_id}_{pol}_temp.vrt'
            burst.slc_to_vrt_file(temp_slc_path)
            rdr_burst_raster = isce3.io.Raster(temp_slc_path)

        # Generate output geocoded burst raster
        geo_burst_raster = isce3.io.Raster(
            f'{cfg.output_dir}/{cfg.file_stem}',
            geo_grid.width, geo_grid.length,
            rdr_burst_raster.num_bands, gdal.GDT_CFloat32,
            cfg.geocoding_params.output_format)

        # Extract burst boundaries
        b_bounds = np.s_[burst.first_valid_line:burst.last_valid_line,
                   burst.first_valid_sample:burst.last_valid_sample]

        # Create sliced radar grid representing valid region of the burst
        sliced_radar_grid = burst.as_isce3_radargrid()[b_bounds]

        # Geocode
        isce3.geocode.geocode_slc(geo_burst_raster, rdr_burst_raster,
                                  dem_raster,
                                  radar_grid, sliced_radar_grid,
                                  geo_grid, orbit,
                                  native_doppler,
                                  image_grid_doppler, ellipsoid, threshold,
                                  iters,
                                  blocksize, dem_margin, flatten,
                                  azimuth_carrier=az_carrier_poly2d)

        # Set geo transformation
        geotransform = [geo_grid.start_x, geo_grid.spacing_x, 0,
                        geo_grid.start_y, 0, geo_grid.spacing_y]
        geo_burst_raster.set_geotransform(geotransform)
        geo_burst_raster.set_epsg(epsg)
        del geo_burst_raster

    dt = time.time() - t_start
    info_channel.log(f'geocode burst successfully ran in {dt:.3f} seconds')


if __name__ == "__main__":
    '''Run geocode cslc workflow from command line'''
    # load arguments from command line
    geo_parser = YamlArgparse()

    # Get a runconfig dict from command line argumens
    cfg = GeoRunConfig.load_from_yaml(geo_parser.run_config_path,
                                      'geo_cslc_s1')

    # Run geocode burst workflow
    run(cfg)

    # Save burst metadata and runconfig parameters
    json_path = f'{cfg.output_dir}/{cfg.file_stem}.json'
    with open(json_path, 'w') as f_json:
        cfg.to_metadata_file(f_json, 'json')

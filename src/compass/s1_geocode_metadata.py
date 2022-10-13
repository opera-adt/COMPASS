#!/usr/bin/env python

'''wrapper to geocode metadata layers'''

from datetime import timedelta
import os
import time

import isce3
import journal

from osgeo import gdal
from compass.utils.runconfig import RunConfig
from compass.utils.helpers import get_module_name
from compass.utils.yaml_argparse import YamlArgparse


def run(cfg, fetch_from_scratch=False):
    '''
    Geocode metadata layers

    Parameters
    ----------
    cfg: dict
        Dictionary with user runconfig options
    fetch_from_scratch: bool
        If True grabs metadata layers from scratch dir

    '''
    module_name = get_module_name(__file__)
    info_channel = journal.info(f"{module_name}.run")
    info_channel.log(f"Starting {module_name} burst")

    # Start tracking processing time
    t_start = time.time()

    # common initializations
    dem_raster = isce3.io.Raster(cfg.dem)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid
    image_grid_doppler = isce3.core.LUT2d()
    threshold = cfg.geo2rdr_params.threshold
    iters = cfg.geo2rdr_params.numiter
    blocksize = cfg.geo2rdr_params.lines_per_block
    output_epsg = cfg.geocoding_params.output_epsg
    output_format = cfg.geocoding_params.output_format

    # process one burst only
    burst = cfg.bursts[0]
    date_str = burst.sensing_start.strftime("%Y%m%d")
    burst_id = burst.burst_id
    geo_grid = cfg.geogrids[burst_id]

    os.makedirs(cfg.output_dir, exist_ok=True)

    scratch_path = f'{cfg.scratch_path}/{burst_id}/{date_str}'
    os.makedirs(scratch_path, exist_ok=True)

    radar_grid = burst.as_isce3_radargrid()
    orbit = burst.orbit

    # Initialize input/output path
    input_path = f'{cfg.product_path}/{burst_id}/{date_str}'
    output_path = input_path
    if fetch_from_scratch:
        input_path = f'{cfg.scratch_path}/{burst_id}/{date_str}'
    os.makedirs(output_path, exist_ok=True)

    # Initialize geocode object
    geo = isce3.geocode.GeocodeFloat32()
    geo.orbit = orbit
    geo.ellipsoid = ellipsoid
    geo.doppler = image_grid_doppler
    geo.threshold_geo2rdr = threshold
    geo.numiter_geo2rdr = iters
    geo.lines_per_block = blocksize
    geo.geogrid(geo_grid.start_x, geo_grid.start_y,
                geo_grid.spacing_x, geo_grid.spacing_y,
                geo_grid.width, geo_grid.length, geo_grid.epsg)

    # Get the metadata layers to compute
    meta_layers = {'x': cfg.rdr2geo_params.compute_longitude,
                   'y': cfg.rdr2geo_params.compute_latitude,
                   'z': cfg.rdr2geo_params.compute_height,
                   'incidence': cfg.rdr2geo_params.compute_incidence_angle,
                   'localIncidence': cfg.rdr2geo_params.compute_local_incidence_angle,
                   'heading': cfg.rdr2geo_params.compute_azimuth_angle}
    input_rasters = [
        isce3.io.Raster(f'{input_path}/{fname}.rdr')
        for fname, enabled in meta_layers.items() if enabled]
    output_rasters = [
        isce3.io.Raster(f'{output_path}/{fname}.geo',
                        geo_grid.width, geo_grid.length, 1, gdal.GDT_Float32,
                        output_format)
        for fname, enabled in meta_layers.items() if enabled]

    # Geocode list of products
    geotransform = [geo_grid.start_x, geo_grid.spacing_x, 0,
                    geo_grid.start_y, 0, geo_grid.spacing_y]
    for input_raster, output_raster in zip(input_rasters, output_rasters):
        geo.geocode(radar_grid=radar_grid, input_raster=input_raster,
                    output_raster=output_raster, dem_raster=dem_raster,
                    output_mode=isce3.geocode.GeocodeOutputMode.INTERP)
        output_raster.set_geotransform(geotransform)
        output_raster.set_epsg(output_epsg)
        del output_raster

    # Geocode layover shadow separately
    if cfg.rdr2geo_params.compute_layover_shadow_mask:
        input_raster = isce3.io.Raster(f'{input_path}/layoverShadowMask.rdr')
        output_raster = isce3.io.Raster(f'{output_path}/layoverShadowMask.geo',
                                        geo_grid.width, geo_grid.length, 1,
                                        gdal.GDT_Byte, output_format)
        geo.data_interpolator = 'NEAREST'
        geo.geocode(radar_grid=radar_grid, input_raster=input_raster,
                    output_raster=output_raster, dem_raster=dem_raster,
                    output_mode=isce3.geocode.GeocodeOutputMode.INTERP)
        output_raster.set_geotransform(geotransform)
        output_raster.set_epsg(output_epsg)
        del output_raster

    dt = str(timedelta(seconds=time.time() - t_start)).split(".")[0]
    info_channel.log(
        f"{module_name} burst successfully ran in {dt} (hr:min:sec)")


if __name__ == "__main__":
    ''' run geocode metadata layers from command line'''
    parser = YamlArgparse()

    # Get a runconfig dict from command line args
    cfg = RunConfig.load_from_yaml(parser.args.run_config_path,
                                   workflow_name='s1_cslc_radar')
    # run geocode metadata layers
    run(cfg)

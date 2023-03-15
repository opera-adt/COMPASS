#!/usr/bin/env python

'''wrapper to geocode metadata layers'''

from datetime import timedelta
import os
import time

import h5py
import isce3
import journal
import numpy as np

from osgeo import gdal
from compass.utils.runconfig import RunConfig
from compass.utils.helpers import get_module_name
from compass.utils.yaml_argparse import YamlArgparse


def run(cfg, burst, fetch_from_scratch=False):
    '''
    Geocode metadata layers in single HDF5

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
    lines_per_block = cfg.geo2rdr_params.lines_per_block
    output_format = cfg.geocoding_params.output_format

    # process one burst only
    date_str = burst.sensing_start.strftime("%Y%m%d")
    burst_id = str(burst.burst_id)
    geo_grid = cfg.geogrids[burst_id]
    output_epsg = geo_grid.epsg

    radar_grid = burst.as_isce3_radargrid()
    orbit = burst.orbit

    # Initialize input/output paths
    burst_id_date_key = (burst_id, date_str)
    out_paths = cfg.output_paths[burst_id_date_key]

    input_path = out_paths.output_directory
    if fetch_from_scratch:
        input_path = out_paths.scratch_directory

    # Initialize geocode object
    geo = isce3.geocode.GeocodeFloat32()
    geo.orbit = orbit
    geo.ellipsoid = ellipsoid
    geo.doppler = image_grid_doppler
    geo.threshold_geo2rdr = threshold
    geo.numiter_geo2rdr = iters
    elements_in_block = lines_per_block * geogrid.width
    geo.geogrid(geo_grid.start_x, geo_grid.start_y,
                geo_grid.spacing_x, geo_grid.spacing_y,
                geo_grid.width, geo_grid.length, geo_grid.epsg)

    # Geocode list of products
    geotransform = [geo_grid.start_x, geo_grid.spacing_x, 0,
                    geo_grid.start_y, 0, geo_grid.spacing_y]

    # Get the metadata layers to compute
    float_bytes = 4
    meta_layers = {'x': (cfg.rdr2geo_params.compute_longitude, float_bytes),
                   'y': (cfg.rdr2geo_params.compute_latitude, float_bytes),
                   'z': (cfg.rdr2geo_params.compute_height, float_bytes),
                   'incidence': (cfg.rdr2geo_params.compute_incidence_angle,
                                 float_bytes),
                   'local_incidence': (cfg.rdr2geo_params.compute_local_incidence_angle,
                                       float_bytes),
                   'heading': (cfg.rdr2geo_params.compute_azimuth_angle,
                               float_bytes),
                   'layover_shadow_mask': (cfg.rdr2geo_params.compute_layover_shadow_mask,
                                           float_bytes)}

    out_h5 = f'{out_paths.output_directory}/topo.h5'
    shape = (geo_grid.length, geo_grid.width)
    with h5py.File(out_h5, 'w') as topo_h5:
        for layer_name, (enabled, type_bytes) in meta_layers.items():
            if not enabled:
                continue
            dtype = np.single
            # layoverShadowMask is last option, no need to change data type
            # and interpolator afterwards
            if layer_name == 'layover_shadow_mask':
                geo.data_interpolator = 'NEAREST'
                dtype = np.byte

            topo_ds = topo_h5.create_dataset(layer_name, dtype=dtype,
                                             shape=shape)
            topo_ds.attrs['description'] = np.string_(layer_name)
            output_raster = isce3.io.Raster(f"IH5:::ID={topo_ds.id.id}".encode("utf-8"),
                                            update=True)

            input_raster = isce3.io.Raster(f'{input_path}/{layer_name}.rdr')

            block_size = elements_in_block * type_bytes
            geo.geocode(radar_grid=radar_grid, input_raster=input_raster,
                        output_raster=output_raster, dem_raster=dem_raster,
                        output_mode=isce3.geocode.GeocodeOutputMode.INTERP,
                        min_block_size=block_size, max_block_size=block_size)
            output_raster.set_geotransform(geotransform)
            output_raster.set_epsg(output_epsg)
            del input_raster
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

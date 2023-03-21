#!/usr/bin/env python

'''wrapper to geocode metadata layers'''

from datetime import timedelta
import time

import h5py
import isce3
import journal
import numpy as np

from compass import s1_rdr2geo
from compass.utils.h5_helpers import init_geocoded_dataset
from compass.utils.helpers import bursts_grouping_generator, get_module_name
from compass.utils.geo_runconfig import GeoRunConfig
from compass.utils.yaml_argparse import YamlArgparse


def run(cfg, burst, fetch_from_scratch=False):
    '''
    Geocode metadata layers in single HDF5

    Parameters
    ----------
    cfg: GeoRunConfig
        GeoRunConfig object with user runconfig options
    burst: Sentinel1BurstSlc
        Object containing burst parameters needed for geocoding
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
    geocode_obj = isce3.geocode.GeocodeFloat32()
    geocode_obj.orbit = orbit
    geocode_obj.ellipsoid = ellipsoid
    geocode_obj.doppler = image_grid_doppler
    geocode_obj.threshold_geo2rdr = threshold
    geocode_obj.numiter_geo2rdr = iters
    float_bytes = 4
    block_size = lines_per_block * geo_grid.width * float_bytes
    geocode_obj.geogrid(geo_grid.start_x, geo_grid.start_y,
                geo_grid.spacing_x, geo_grid.spacing_y,
                geo_grid.width, geo_grid.length, geo_grid.epsg)

    # Geocode list of products
    geotransform = [geo_grid.start_x, geo_grid.spacing_x, 0,
                    geo_grid.start_y, 0, geo_grid.spacing_y]

    # Get the metadata layers to compute
    meta_layers = {'x': cfg.rdr2geo_params.compute_longitude,
                   'y': cfg.rdr2geo_params.compute_latitude,
                   'z': cfg.rdr2geo_params.compute_height,
                   'incidence': cfg.rdr2geo_params.compute_incidence_angle,
                   'local_incidence': cfg.rdr2geo_params.compute_local_incidence_angle,
                   'heading': cfg.rdr2geo_params.compute_azimuth_angle,
                   'layover_shadow_mask': cfg.rdr2geo_params.compute_layover_shadow_mask
                   }

    out_h5 = f'{out_paths.output_directory}/topo.h5'
    with h5py.File(out_h5, 'w') as topo_root_h5:
        # Geocode designated layers
        for layer_name, enabled in meta_layers.items():
            if not enabled:
                continue

            dtype = np.single
            # layoverShadowMask is last option, no need to change data type
            # and interpolator afterwards
            if layer_name == 'layover_shadow_mask':
                geocode_obj.data_interpolator = 'NEAREST'
                dtype = np.byte

            # Create dataset with x/y coords/spacing and projection
            topo_ds = init_geocoded_dataset(topo_root_h5, layer_name, geo_grid,
                                            dtype, np.string_(layer_name))

            # Init output and input isce3.io.Raster objects for geocoding
            output_raster = isce3.io.Raster(f"IH5:::ID={topo_ds.id.id}".encode("utf-8"),
                                            update=True)

            input_raster = isce3.io.Raster(f'{input_path}/{layer_name}.rdr')

            geocode_obj.geocode(radar_grid=radar_grid,
                                input_raster=input_raster,
                                output_raster=output_raster,
                                dem_raster=dem_raster,
                                output_mode=isce3.geocode.GeocodeOutputMode.INTERP,
                                min_block_size=block_size,
                                max_block_size=block_size)
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
    cfg = GeoRunConfig.load_from_yaml(parser.args.run_config_path,
                                      workflow_name='s1_cslc_geo')

    for _, bursts in bursts_grouping_generator(cfg.bursts):
        burst = bursts[0]

        # Generate required static layers
        if cfg.rdr2geo_params.enabled:
            s1_rdr2geo.run(cfg, save_in_scratch=True)

            # Geocode static layers if needed
            if cfg.rdr2geo_params.geocode_metadata_layers:
                run(cfg, burst, fetch_from_scratch=True)

#!/usr/bin/env python

'''wrapper for rdr2geo'''

import time

import isce3
import journal
from osgeo import gdal

from compass.utils.helpers import get_module_name, get_time_delta_str
from compass.utils.radar_grid import rdr_grid_to_file
from compass.utils.runconfig import RunConfig
from compass.utils.yaml_argparse import YamlArgparse

file_name_los_east = 'los_east'
file_name_los_north = 'los_north'
file_name_local_incidence = 'local_incidence_angle'
file_name_layover = 'layover_shadow_mask'
file_name_x = 'x'
file_name_y = 'y'
file_name_z = 'z'


def run(cfg, burst=None, save_in_scratch=False):
    '''run rdr2geo with provided runconfig

    Parameters
    ----------
    cfg: dict
        Runconfig dictionary with user-defined options
    burst: Sentinel1BurstSlc
        Burst to run rdr2geo. If `None`, it will process all bursts in `cfg`
    save_in_scratch: bool
        Flag to save output in scratch dir instead of product dir
    '''
    module_name = get_module_name(__file__)
    info_channel = journal.info(f"{module_name}.run")
    info_channel.log(f"Starting {module_name} burst")

    # Tracking time elapsed for processing
    t_start = time.perf_counter()

    # Extract rdr2geo cfg
    rdr2geo_cfg = cfg.rdr2geo_params

    # common rdr2geo inits
    dem_raster = isce3.io.Raster(cfg.dem)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # check if gpu ok to use and init CPU or CUDA object accordingly
    use_gpu = isce3.core.gpu_check.use_gpu(cfg.gpu_enabled, cfg.gpu_id)
    if use_gpu:
        # Set the current CUDA device.
        device = isce3.cuda.core.Device(cfg.gpu_id)
        isce3.cuda.core.set_device(device)
        Rdr2Geo = isce3.cuda.geometry.Rdr2Geo
    else:
        Rdr2Geo = isce3.geometry.Rdr2Geo

    # list to keep track of ids processed
    id_processed = []

    # run rdr2geo for only once per burst_id
    # save SLC for all bursts
    if burst is None:
        bursts = cfg.bursts
    else:
        bursts = [burst]

    for burst in bursts:
        # extract date string and create directory
        date_str = burst.sensing_start.strftime("%Y%m%d")
        burst_id = str(burst.burst_id)

        # init output directory in product_path
        burst_id_date_key = (burst_id, date_str)
        out_paths = cfg.output_paths[burst_id_date_key]
        output_path = out_paths.output_directory
        if save_in_scratch:
            output_path = out_paths.scratch_directory

        # save SLC to Geotiff for all bursts
        # run rdr2geo for only 1 burst avoid redundancy
        burst.slc_to_file(f'{output_path}/{out_paths.file_name_pol}.slc.tif', 'GTiff')

        # skip burst if id already rdr2geo processed
        # save id if not processed to avoid rdr2geo reprocessing
        if burst_id in id_processed:
            continue
        id_processed.append(burst_id)

        # get radar grid of last SLC written and save for resample flattening
        rdr_grid = burst.as_isce3_radargrid()
        ref_grid_path = f'{output_path}/radar_grid.txt'
        rdr_grid_to_file(ref_grid_path, rdr_grid)

        # get isce3 objs from burst
        isce3_orbit = burst.orbit

        # init grid doppler
        grid_doppler = isce3.core.LUT2d()

        # init rdr2geo obj
        rdr2geo_obj = Rdr2Geo(rdr_grid, isce3_orbit, ellipsoid, grid_doppler,
                              threshold=rdr2geo_cfg.threshold,
                              numiter=rdr2geo_cfg.numiter,
                              extraiter=rdr2geo_cfg.extraiter,
                              lines_per_block=rdr2geo_cfg.lines_per_block)

        # Dict containing the rdr2geo layers to generate and their filenames
        # key: rdr2geo layer name
        # value: (boolean flag; True if layers needs to be generated, layer name)
        topo_output = {file_name_x: (rdr2geo_cfg.compute_longitude, gdal.GDT_Float64),
                       file_name_y: (rdr2geo_cfg.compute_latitude, gdal.GDT_Float64),
                       file_name_z: (rdr2geo_cfg.compute_height, gdal.GDT_Float64),
                       file_name_layover: (
                       cfg.rdr2geo_params.compute_layover_shadow_mask,
                       gdal.GDT_Byte),
                       file_name_local_incidence: (
                       rdr2geo_cfg.compute_local_incidence_angle,
                       gdal.GDT_Float32),
                       file_name_los_east: (
                       rdr2geo_cfg.compute_ground_to_sat_east, gdal.GDT_Float32),
                       file_name_los_north: (
                       rdr2geo_cfg.compute_ground_to_sat_north, gdal.GDT_Float32),
                       }
        raster_list = [
            isce3.io.Raster(f'{output_path}/{fname}.rdr', rdr_grid.width,
                            rdr_grid.length, 1, dtype, 'GTiff')
            if enabled else None
            for fname, (enabled, dtype) in topo_output.items()]

        (x_raster, y_raster, z_raster, layover_shadow_raster,
         local_incident_angle_raster, los_east_raster,
         los_north_raster) = raster_list

        # run rdr2geo
        rdr2geo_obj.topo(dem_raster, x_raster=x_raster, y_raster=y_raster,
                         height_raster=z_raster,
                         local_incidence_angle_raster=local_incident_angle_raster,
                         layover_shadow_raster=layover_shadow_raster,
                         ground_to_sat_east_raster=los_east_raster,
                         ground_to_sat_north_raster=los_north_raster)

        # remove undesired/None rasters from raster list
        raster_list = [raster for raster in raster_list if raster is not None]

        # save non-None rasters to vrt
        output_vrt = isce3.io.Raster(f'{output_path}/topo.vrt', raster_list)
        output_vrt.set_epsg(rdr2geo_obj.epsg_out)

    dt = get_time_delta_str(t_start)
    info_channel.log(
        f"{module_name} burst successfully ran in {dt} (hr:min:sec)")


if __name__ == "__main__":
    '''run rdr2geo from command line'''
    # load command line args
    parser = YamlArgparse()

    # get a runconfig dict from command line args
    cfg = RunConfig.load_from_yaml(parser.args.run_config_path,
                                   workflow_name='s1_cslc_radar')

    # run rdr2geo
    run(cfg)

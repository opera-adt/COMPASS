#!/usr/bin/env python3

'''wrapper for rdr2geo'''

import os
import time

import isce3
import journal
from osgeo import gdal

from compass.utils.logger import Logger
from compass.utils.runconfig import RunConfig
from compass.utils.yaml_argparse import YamlArgparse


def run(cfg):
    '''run rdr2geo with provided runconfig'''
    info_channel = journal.info("rdr2geo.run")
    logging = Logger(channel=info_channel, workflow='CSLC')

    t_start = time.time()
    logging.info('rdr2geo.py', 9999, "starting rdr2geo")

    # common rdr2geo inits
    dem_raster = isce3.io.Raster(cfg.dem)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # check if gpu ok to use
    use_gpu = isce3.core.gpu_check.use_gpu(cfg.gpu_enabled, cfg.gpu_id)
    if use_gpu:
        # Set the current CUDA device.
        device = isce3.cuda.core.Device(cfg.gpu_id)
        isce3.cuda.core.set_device(device)

    # save SLC for all bursts
    # run rdr2geo for only once per burst_id
    for bursts in cfg.bursts:
        # init output directory in product_path
        output_path = f'{cfg.product_path}/{bursts[0].burst_id}'
        os.makedirs(output_path, exist_ok=True)

        # save SLC to ENVI for all bursts
        for burst in bursts:
            burst.slc_to_file(f'{output_path}/{burst.polarization}.slc')

        # run rdr2geo for only 1 burst avoid redundancy
        # get isce3 objs from burst
        rdr_grid = burst.as_isce3_radargrid()
        isce3_orbit = burst.orbit

        # init grid doppler
        grid_doppler = isce3.core.LUT2d()

        # init CPU or CUDA object accordingly
        if use_gpu:
            Rdr2Geo = isce3.cuda.geometry.Rdr2Geo
        else:
            Rdr2Geo = isce3.geometry.Rdr2Geo

        # init rdr2geo obj
        rdr2geo_obj = Rdr2Geo(
            rdr_grid,
            isce3_orbit,
            ellipsoid,
            grid_doppler)

        # set rdr2geo params
        for key, val in cfg.rdr2geo_params.__dict__.items():
            setattr(rdr2geo_obj, key, val)

        # prepare output rasters
        topo_output = {'x':(True, gdal.GDT_Float64),
                       'y':(True, gdal.GDT_Float64),
                       'z':(True, gdal.GDT_Float64),
                       'layoverShadowMask':(cfg.rdr2geo_params.compute_mask,
                                            gdal.GDT_Byte)}
        raster_list = [
            isce3.io.Raster(f'{output_path}/{fname}.rdr', rdr_grid.width,
                            rdr_grid.length, 1, dtype, 'ENVI')
            if enabled else None
            for fname, (enabled, dtype) in topo_output.items()]
        x_raster, y_raster, z_raster, layover_shadow_raster = raster_list

        # run rdr2geo
        rdr2geo_obj.topo(dem_raster, x_raster, y_raster, z_raster,
                         layover_shadow_raster=layover_shadow_raster)

        # remove undesired/None rasters from raster list
        raster_list = [raster for raster in raster_list if raster is not None]

        # save non-None rasters to vrt
        output_vrt = isce3.io.Raster(f'{output_path}/topo.vrt', raster_list)
        output_vrt.set_epsg(rdr2geo_obj.epsg_out)

    dt = time.time() - t_start
    logging.info("rdr2geo.py", 9999, f"rdr2geo successfully ran in {dt:.3f} seconds")


if __name__ == "__main__":
    '''run rdr2geo from command line'''
    # load command line args
    rdr2geo_parser = YamlArgparse()

    # get a runconfig dict from command line args
    rdr2geo_runconfig = RunConfig.load_from_yaml(rdr2geo_parser.args.run_config_path, 'rdr2geo')

    # run rdr2geo
    run(rdr2geo_runconfig)

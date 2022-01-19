#!/usr/bin/env python3

'''wrapper for rdr2geo'''

import os
import time

import isce3
import journal

from compass.utils.runconfig import RunConfig
from compass.utils.yaml_argparse import YamlArgparse

def run(cfg):
    '''run rdr2geo with provided runconfig'''
    info_channel = journal.info("rdr2geo.run")

    t_start = time.time()
    info_channel.log("starting rdr2geo")

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

    for burst in cfg.bursts:
        # get isce3 objs from burst
        rdr_grid = burst.as_isce3_radargrid()
        isce3_orbit = burst.orbit

        # init output directory in scratch
        output_path = f'{cfg.scratch_path}/{burst.burst_id}'
        os.makedirs(output_path, exist_ok=True)

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

        # turn off shadow layover mask
        rdr2geo_obj.compute_mask = False

        # set rdr2geo params
        for key, val in cfg.rdr2geo_params.__dict__.items():
            setattr(rdr2geo_obj, key, val)

        # run rdr2geo
        rdr2geo_obj.topo(dem_raster, output_path)

    dt = time.time() - t_start
    info_channel.log(f"rdr2geo successfully ran in {dt:.3f} seconds")


if __name__ == "__main__":
    '''run rdr2geo from command line'''
    # load command line args
    rdr2geo_parser = YamlArgparse()

    # get a runconfig dict from command line args
    rdr2geo_runconfig = RunConfig.load_from_yaml(rdr2geo_parser.args.run_config_path, 'rdr2geo')

    # run rdr2geo
    run(rdr2geo_runconfig)

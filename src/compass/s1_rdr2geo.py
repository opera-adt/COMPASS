#!/usr/bin/env python3

'''wrapper for rdr2geo'''

from datetime import timedelta
import os
import time

import isce3
import journal
from osgeo import gdal

from compass.utils.radar_grid import rdr_grid_to_file
from compass.utils.runconfig import RunConfig
from compass.utils.yaml_argparse import YamlArgparse


def run(cfg):
    '''run rdr2geo with provided runconfig'''
    info_channel = journal.info("s1_rdr2geo.run")

    t_start = time.time()
    info_channel.log("starting rdr2geo")

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
    for burst in cfg.bursts:
        # extract date string and create directory
        date_str = str(burst.sensing_start.date())
        burst_id = burst.burst_id

        # init output directory in product_path
        output_path = f'{cfg.product_path}/{burst_id}/{date_str}'
        os.makedirs(output_path, exist_ok=True)

        # save SLC to ENVI for all bursts
        # run rdr2geo for only 1 burst avoid redundancy
        burst.slc_to_file(f'{output_path}/{burst.polarization}.slc')

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

    dt = str(timedelta(seconds=time.time() - t_start))
    info_channel.log(f"rdr2geo successfully ran in {dt} (hr:min:sec)")


if __name__ == "__main__":
    '''run rdr2geo from command line'''
    # load command line args
    rdr2geo_parser = YamlArgparse()

    # get a runconfig dict from command line args
    rdr2geo_runconfig = RunConfig.load_from_yaml(rdr2geo_parser.args.run_config_path, 'rdr2geo')

    # run rdr2geo
    run(rdr2geo_runconfig)

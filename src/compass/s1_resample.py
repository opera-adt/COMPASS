#!/usr/bin/env python

"""Wrapper for resample"""

import os
import time

import isce3
import journal
from osgeo import gdal

from compass.utils.helpers import get_module_name, get_time_delta_str
from compass.utils.runconfig import RunConfig
from compass.utils.yaml_argparse import YamlArgparse


def run(cfg: dict):
    """
    Run resample burst with user-defined options

    Parameters
    ----------
    cfg: dict
        Runconfig dictionary with user-defined options
    """
    module_name = get_module_name(__file__)
    info_channel = journal.info(f"{module_name}.run")
    info_channel.log(f"Starting {module_name} burst")

    # Tracking time elapsed for processing
    t_start = time.perf_counter()

    # Check if user wants to use GPU for processing
    # Instantiate and initialize resample object
    use_gpu = isce3.core.gpu_check.use_gpu(cfg.gpu_enabled, cfg.gpu_id)
    if use_gpu:
        device = isce3.cuda.core.Device(cfg.gpu_id)
        isce3.cuda.core.set_device(device)
        resamp = isce3.cuda.image.ResampSlc
    else:
        resamp = isce3.image.ResampSlc

    # Get common resample parameters
    blocksize = cfg.resample_params.lines_per_block

    # Process all bursts
    for burst in cfg.bursts:
        # get burst ID and date string of current burst
        burst_id = str(burst.burst_id)
        date_str = burst.sensing_start.strftime("%Y%m%d")

        # Create top output path
        burst_id_date_key = (burst_id, date_str)
        out_paths = cfg.output_paths[burst_id_date_key]

        # Get reference burst radar grid
        ref_rdr_grid = cfg.reference_radar_info.grid

        # Get radar grid
        rdr_grid = burst.as_isce3_radargrid()

        # Extract azimuth carrier polynomials
        az_poly = burst.get_az_carrier_poly()

        # Init resample SLC object
        resamp_obj = resamp(rdr_grid, burst.doppler.lut2d,
                            az_poly, ref_rdr_grid=ref_rdr_grid)
        resamp_obj.lines_per_tile = blocksize

        # Get range and azimuth offsets
        offset_path = out_paths.scratch_directory
        rg_off_raster = isce3.io.Raster(f'{offset_path}/range.off')
        az_off_raster = isce3.io.Raster(f'{offset_path}/azimuth.off')

        # Get original SLC as raster object
        sec_burst_path = f'{out_paths.scratch_directory}/{out_paths.file_name_pol}.slc.vrt'
        burst.slc_to_vrt_file(sec_burst_path)
        original_raster = isce3.io.Raster(sec_burst_path)

        # Prepare resampled SLC as raster object
        coreg_burst_path = f'{out_paths.output_directory}/{out_paths.file_name_stem}.slc.tif'
        resampled_raster = isce3.io.Raster(coreg_burst_path,
                                           rg_off_raster.width,
                                           rg_off_raster.length,
                                           1, gdal.GDT_CFloat32,
                                           'GTiff')

        resamp_obj.resamp(original_raster, resampled_raster,
                          rg_off_raster, az_off_raster,
                          flatten=cfg.resample_params.flatten)

    dt = get_time_delta_str(t_start)
    info_channel.log(f"{module_name} burst successfully ran in {dt} (hr:min:sec)")


if __name__ == "__main__":
    """Run resample burst from command line"""
    parser = YamlArgparse()

    # Get a runconfig dict from command line arguments
    cfg = RunConfig.load_from_yaml(parser.args.run_config_path,
                                   workflow_name='s1_cslc_radar')

    # Run resample burst
    run(cfg)

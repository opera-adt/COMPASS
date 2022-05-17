"""Wrapper for resample burst"""
from datetime import timedelta
import os
import time

import isce3
import journal
from osgeo import gdal

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
    info_channel = journal.info("resample_burst.run")
    info_channel.log("Starting resample burst")

    # Tracking time elapsed for processing
    t_start = time.time()

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
        # get burst ID of current burst
        burst_id = burst.burst_id

        # Create top output path
        top_output_path = f'{cfg.product_path}/{burst_id}'
        os.makedirs(top_output_path, exist_ok=True)

        # Get reference burst radar grid
        ref_rdr_grid = cfg.reference_radar_info.grid

        # Extract date string and create directory
        date_str = str(burst.sensing_start.date())
        burst_output_path = f'{top_output_path}/{date_str}'
        os.makedirs(burst_output_path, exist_ok=True)

        # Extract polarization
        pol = burst.polarization

        # Get radar grid
        rdr_grid = burst.as_isce3_radargrid()

        # Extract azimuth carrier polynomials
        az_poly = burst.get_az_carrier_poly(index_as_coord=True)

        # Init resample SLC object
        resamp_obj = resamp(rdr_grid, burst.doppler.lut2d,
                            az_poly, ref_rdr_grid=ref_rdr_grid)
        resamp_obj.lines_per_tile = blocksize

        # Get range and azimuth offsets
        offset_path = f'{cfg.scratch_path}/' \
                      f'{burst_id}/{date_str}'
        rg_off_raster = isce3.io.Raster(f'{offset_path}/range.off')
        az_off_raster = isce3.io.Raster(f'{offset_path}/azimuth.off')

        # Get original SLC as raster object
        sec_burst_path = f'{cfg.scratch_path}/{burst_id}_{date_str}_{pol}.slc.vrt'
        burst.slc_to_vrt_file(sec_burst_path)
        original_raster = isce3.io.Raster(sec_burst_path)

        # Prepare resamled SLC as raster object
        coreg_burst_path = f'{burst_output_path}/{pol}.slc'
        resampled_raster = isce3.io.Raster(coreg_burst_path,
                                           rg_off_raster.width,
                                           rg_off_raster.length,
                                           1, gdal.GDT_CFloat32,
                                           'ENVI')

        resamp_obj.resamp(original_raster, resampled_raster,
                          rg_off_raster, az_off_raster,
                          flatten=cfg.resample_params.flatten)

    dt = str(timedelta(seconds=time.time() - t_start))
    info_channel.log(f"resample burst successfully ran in {dt} (hr:min:sec)")


if __name__ == "__main__":
    """Run resample burst from command line"""
    parser = YamlArgparse()
    parser.parse()

    # Get a runconfig dict from command line arguments
    resample_runconfig = RunConfig.load_from_yaml(
        resample_parser.args.run_config_path, 'resample_burst')

    # Run resample burst
    run(resample_runconfig)

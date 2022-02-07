"""Wrapper for resample burst"""
import time
import os

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
        device = isce3.core.cuda.Device(cfg.gpu_id)
        isce3.cuda.core.set_device(device)
        resamp = isce3.cuda.image.ResampSlc
    else:
        resamp = isce3.image.ResampSlc

    # Get common resample parameters
    blocksize = cfg.resample_params.lines_per_block

    for bursts in cfg.bursts:
        # Create output path
        output_path = f'{cfg.product_path}/{bursts[0].burst_id}'
        os.makedirs(output_path, exist_ok=True)

        for burst in bursts:
            # Get radar grid
            rdr_grid = burst.as_isce3_radargrid()

            # Extract azimuth carrier polynomials
            az_poly = burst.get_az_carrier_poly(index_as_coord=True)

            resamp_obj = resamp(rdr_grid, burst.doppler.lut2d,
                                az_poly)
            resamp_obj.lines_per_tile = blocksize

            # Get range and azimuth offsets
            offset_path = f'{cfg.scratch_path}/' \
                          f'{burst.burst_id}/geo2rdr'
            rg_off_raster = isce3.io.Raster(f'{offset_path}/range.off')
            az_off_raster = isce3.io.Raster(f'{offset_path}/azimuth.off')

            # Extract polarization
            pol = burst.polarization
            sec_burst_path = f'{cfg.scratch_path}/{burst.burst_id}/sec_burst_{pol}.slc'
            burst.slc_to_file(sec_burst_path)
            burst_raster = isce3.io.Raster(sec_burst_path)
            coreg_burst_path = f'{output_path}/coreg_{burst.burst_id}_{pol}.slc'
            resamp_burst_raster = isce3.io.Raster(coreg_burst_path,
                                                  rg_off_raster.width,
                                                  rg_off_raster.length,
                                                  rg_off_raster.num_bands,
                                                  gdal.GDT_CFloat32,
                                                  'ENVI')
            resamp_obj.resamp(burst_raster, resamp_burst_raster,
                              rg_off_raster, az_off_raster)
    dt = time.time() - t_start
    info_channel.log(f"resample burst successfully ran in {dt:.3f} seconds")


if __name__ == "__main__":
    """Run resample burst from command line"""
    resample_parser = YamlArgparse()

    # Get a runconfig dict from command line arguments
    resample_runconfig = RunConfig.load_from_yaml(
        resample_parser.args.run_config_path, 'resample_burst')

    # Run resample burst
    run(resample_runconfig)

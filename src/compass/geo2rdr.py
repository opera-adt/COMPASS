"""Wrapper for geo2rdr"""
import os
import time

import isce3
import journal

from compass.utils.runconfig import RunConfig
from compass.utils.yaml_argparse import YamlArgparse


def run(cfg: dict):
    """
    Run geo2rdr with user-defined options
    stored in runconfig dictionary (cfg)

    Parameters
    ----------
    cfg: dict
      Dictionary with user-defined options
    """
    info_channel = journal.info("geo2rdr.run")
    info_channel.log('Starting geo2rdr')

    # Tracking time elapsed for processing
    t_start = time.time()

    # Common initializations for different bursts
    dem_raster = isce3.io.Raster(cfg.dem)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # Check if user wants to use GPU for processing
    # Initialize CPU or GPU geo2rdr object
    use_gpu = isce3.core.gpu_check.use_gpu(cfg.gpu_enabled, cfg.gpu_id)
    if use_gpu:
        device = isce3.cuda.core.Device(cfg.gpu_id)
        isce3.cuda.core.set_device(device)
        geo2rdr = isce3.cuda.geometry.Geo2Rdr
    else:
        geo2rdr = isce3.geometry.Geo2Rdr

    # Get specific geo2rdr parameters from runconfig
    threshold = cfg.geo2rdr_params.threshold
    iters = cfg.geo2rdr_params.numiter
    blocksize = cfg.geo2rdr_params.lines_per_block

    # Process all bursts
    for bursts in cfg.bursts:
        # Create top output path
        top_output_path = f'{cfg.scratch_path}/{bursts[0].burst_id}'
        os.makedirs(top_output_path, exist_ok=True)

        # Keep track of dates processed
        dates_processed = []

        # Process bursts with same burst ID
        for burst in bursts:
            # Extract date string and create directory
            date_str = str(burst.sensing_start.date())

            # If date has been processed, do not re-run geo2rdr
            # This avoid geo2rdr reprocessing for dual-pol
            if date_str in dates_processed:
                continue
            dates_processed.append(date_str)

            # Create date directory
            burst_output_path = f'{top_output_path}/{date_str}'
            os.makedirs(burst_output_path, exist_ok=True)

            # Get topo layers
            input_path = f'{cfg.reference_path}/{burst.burst_id}'
            topo_raster = isce3.io.Raster(f'{input_path}/topo.vrt')

            # Get radar grid and orbit
            rdr_grid = burst.as_isce3_radargrid()
            orbit = burst.orbit

            # Initialize geo2rdr object
            geo2rdr_obj = geo2rdr(rdr_grid, orbit, ellipsoid,
                                  isce3.core.LUT2d(),
                                  threshold, iters,
                                  blocksize)

            # Execute geo2rdr
            geo2rdr_obj.geo2rdr(topo_raster, burst_output_path)

    dt = time.time() - t_start
    info_channel.log(f"geo2rdr successfully ran in {dt:.3f} seconds")


if __name__ == "__main__":
    """Run geo2rdr from command line"""
    geo2rdr_parser = YamlArgparse()

    # Get a runconfig dict from command line arguments
    geo2rdr_runconfig = RunConfig.load_from_yaml(
        geo2rdr_parser.args.run_config_path, 'geo2rdr')

    # Run geo2rdr
    run(geo2rdr_runconfig)

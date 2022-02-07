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
        device = isce3.core.cuda.Device(cfg.gpu_id)
        isce3.cuda.core.set_device(device)
        geo2rdr = isce3.cuda.geometry.Geo2Rdr
    else:
        geo2rdr = isce3.geometry.Geo2Rdr

    # Get specific geo2rdr parameters from runconfig
    threshold = cfg.geo2rdr_params.__dict__["threshold"]
    iters = cfg.geo2rdr_params.__dict__["numiter"]
    blocksize = cfg.geo2rdr_params.__dict__["lines_per_block"]

    for file_path,bursts in zip(cfg.reference_path, cfg.bursts):
        # Create output directory
        output_path = f'{cfg.scratch_path}/' \
                      f'{bursts[0].burst_id}/geo2rdr'
        os.makedirs(output_path, exist_ok=True)

        # Get topo layers
        topo_raster = isce3.io.Raster(f'{file_path}/topo.vrt')

        # Get radar grid and orbit for the burst
        for burst in bursts:
            rdr_grid = burst.as_isce3_radargrid()
            orbit = burst.orbit

            # Initialize geo2rdr object
            geo2rdr_obj = geo2rdr(rdr_grid, orbit, ellipsoid,
                                  isce3.core.LUT2d(),
                                  threshold, iters,
                                  blocksize)
            # Execute geo2rdr
            geo2rdr_obj.geo2rdr(topo_raster, output_path)

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

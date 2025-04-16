#!/usr/bin/env python

"""wrapper for geo2rdr"""

import os
import time

import isce3
import journal

from compass.utils.helpers import get_module_name, get_time_delta_str
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
    module_name = get_module_name(__file__)
    info_channel = journal.info(f"{module_name}.run")
    info_channel.log(f"Starting {module_name} burst")

    # Tracking time elapsed for processing
    t_start = time.perf_counter()

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

    # list to keep track of id+dates pairs processed
    id_dates_processed = []

    # Run geo2rdr once per burst ID + date pair
    for burst in cfg.bursts:
        # Extract date string and create directory
        burst_id = str(burst.burst_id)
        date_str = burst.sensing_start.strftime("%Y%m%d")
        id_date = (burst_id, date_str)
        out_paths = cfg.output_paths[id_date]

        # This ensures running geo2rdr only once; avoiding running for the different polarizations of the same burst_id
        if id_date in id_dates_processed:
            continue
        id_dates_processed.append(id_date)

        # Get topo layers from vrt
        ref_burst_path = cfg.reference_radar_info.path
        topo_raster = isce3.io.Raster(f"{ref_burst_path}/topo.vrt")

        # Get radar grid and orbit
        rdr_grid = burst.as_isce3_radargrid()
        orbit = burst.orbit

        # Initialize geo2rdr object
        geo2rdr_obj = geo2rdr(
            rdr_grid, orbit, ellipsoid, isce3.core.LUT2d(), threshold, iters, blocksize
        )

        # Execute geo2rdr
        geo2rdr_obj.geo2rdr(topo_raster, out_paths.output_directory)

    dt = get_time_delta_str(t_start)
    info_channel.log(f"{module_name} burst successfully ran in {dt} (hr:min:sec)")


if __name__ == "__main__":
    """Run geo2rdr from command line"""
    parser = YamlArgparse()

    # Get a runconfig dict from command line arguments
    cfg = RunConfig.load_from_yaml(
        parser.args.run_config_path, workflow_name="s1_cslc_radar"
    )

    # Run geo2rdr
    run(cfg)

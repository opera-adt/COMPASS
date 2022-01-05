#!/usr/bin/env python3

'''
wrapper for rdr2geo
'''

from itertools import cycle
import logging
import os
import time

import isce3

from runconfig import RunConfig
from yaml_argparse import YamlArgparse


def run(cfg):
    # find bursts identified in cfg.burst_id
    bursts = []
    pols = cfg.polarization
    i_subswaths = [1, 2, 3]
    zip_list = zip(pols, cycle(i_subswaths)) if len(pols) > 3 else zip(
        cyle(pols), i_subswaths)
    for pol, i_subswath in zip_list:
        temp_bursts = [
            b for b in zip2bursts(cfg.safe_file, i_subswath, pol)
            if b.burst_id in cfg.burst_id
        ]
        bursts.extend(temp_bursts)

    if not bursts:
        err_str = "Given burst IDs not found in provided safe file"
        raise ValueError(err_str)

    # common rdr2geo inits
    dem_raster = isce3.io.Raster(cfg.dem)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # process rdr2geo for found bursts
    for i, burst in enumerate(bursts):
        # get isce3 objs from burst
        rdr_grid = burst.as_isce3_radargrid()
        isce3_orbit = burst.get_isce3_orbit(cfg.orbit_path)

        # init output directory
        output_path = f'{cfg.sas_output_file}/{burst.burst_id}'
        os.makedirs(output_path, exist_ok=True)

        # init rdr2geo obj
        rdr2geo_obj = isce3.cuda.geometry.Rdr2Geo(
            rdr_grid,
            isce3_orbit,
            ellipsoid,
            isce3.core.LUT2d())

        # turn off shadow layover mask
        rdr2geo_obj.compute_mask = False

        # set rdr2geo params
        for key, val in cfg.rdr2geo_params:
            setattr(rdr2geo_obj, key, val)

        # run rdr2geo
        rdr2geo_obj.topo(dem_raster, output_path)


if __name__ == "__main__":
    '''run rdr2geo from command line
    '''
    # load command line args
    rdr2geo_parser = YamlArgparse()
    # get a runconfig dict from command line args
    rdr2geo_runconfig = RunConfig(rdr2geo_parser.args)
    # run rdr2geo
    run(rdr2geo_runconfig)

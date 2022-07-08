#!/usr/bin/env python3
import pytest

from compass import s1_rdr2geo
import numpy as np
from osgeo import gdal

import prep

def get_rdr2geo_output(cfg):
    burst = cfg.bursts[0]
    burst_id = burst.burst_id
    date_str = str(burst.sensing_start.date())
    topo_path = f'{cfg.product_path}/{burst_id}/{date_str}/topo.vrt'
    ds = gdal.Open(topo_path)
    llh = np.vstack([ds.GetRasterBand(i+1).ReadAsArray() for i in range(3)])
    return llh.transpose()

def test_45_lat_rdr2geo():
    # set up test configuration for 45 degree latitude
    cfg = prep.cfg_45_lat()

    # run s1_rdr2geo with configuration from above
    s1_rdr2geo.run(cfg)

    # retrieve and print rdr2geo output
    computed_llh = get_rdr2geo_output(cfg)
    print(computed_llh, computed_llh.shape)

    # print analytical solution
    print(cfg.expected_llh, cfg.expected_llh.shape)


if __name__ == '__main__':
    test_45_lat_rdr2geo()

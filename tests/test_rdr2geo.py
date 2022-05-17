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
    print(topo_path)
    ds = gdal.Open(topo_path)
    llh = np.vstack([ds.GetRasterBand(1).ReadAsArray() for i in range(3)])
    return llh

def test_45_lat_rdr2geo():
    cfg = prep.cfg_45_lat()

    expected_llh = prep.compute_expected_llh(cfg.test_params)

    s1_rdr2geo.run(cfg)
    computed_llh = get_rdr2geo_output(cfg)


if __name__ == '__main__':
    test_45_lat_rdr2geo()

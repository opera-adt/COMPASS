#!/usr/bin/env python3
import pytest

from compass import s1_rdr2geo
import prep

def test_45_lat_rdr2geo():
    cfg = prep.cfg_45_lat()
    s1_rdr2geo.run(cfg)


if __name__ == '__main__':
    test_45_lat_rdr2geo()

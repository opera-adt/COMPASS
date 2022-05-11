import datetime
import pytest
from types import SimpleNamespace

import isce3
import numpy as np
from osgeo import gdal
from s1reader.s1_burst_slc import Doppler, Sentinel1BurstSlc

def zero_dem(params):
    lon0 = params.lon0 - params.omega / 2
    lat0 = params.lat0 - params.omega / 2
    geo_trans = [lon0, params.omega, 0.0, lat0, 0.0, params.omega]

    length = 1
    width = params.n_sv + 1

    # create DEM file
    dem_raster = isce3.io.Raster(params.dem, width, length, 1,
                                 gdal.GDT_Float32, 'GTiff')
    dem_raster.set_geotransform(geo_trans)
    dem_raster.set_epsg(params.epsg)
    del dem_raster

    # write zero to DEM file
    data = np.zeros((length, width))
    ds = gdal.Open(params.dem, gdal.GA_Update)
    ds.GetRasterBand(1).WriteArray(data)
    ds = None


def solve_geocentric_lat(params):
    temp = 1 + params.h_sat / params.ellipsoid.a
    temp1 = rng / params.ellipsoid.a
    temp2 = rng / (params.ellipsoid.a + params.h_sat)

    cosang = 0.5 * (temp + (1.0/temp) - temp1 * temp2)
    angdiff = np.arccos(cosang);

    if (params.look_side * satomega > 0):
        x = satlat0 + angdiff
    else:
        x = satlat0 - angdiff

    return x


def compute_orbit(params):
    # Setup orbit
    statevecs = []
    clat = np.cos(params.lat0)
    slat = np.sin(params.lat0)
    sat_h = params.ellipsoid.a + params.h_sat
    for ii in range(params.n_sv):
        total_delta_t = ii * params.orbit_dt
        t = params.sensing_start + datetime.timedelta(seconds=total_delta_t)

        lon = params.lon0 + params.omega * total_delta_t

        pos = [sat_h * clat * np.cos(lon),
                sat_h * clat * np.sin(lon),
                sat_h * slat]

        vel = [-params.omega * pos[1],
                params.omega * pos[0],
                0.0]

        sv = isce3.core.StateVector(isce3.core.DateTime(t), pos, vel)
        statevecs.append(sv)

    # use list of stateVectors to init and return isce3.core.Orbit
    time_delta = datetime.timedelta(days=2)
    ref_epoch = isce3.core.DateTime(params.sensing_start - time_delta)

    return isce3.core.Orbit(statevecs, ref_epoch)


def common_params(lat):
    params = SimpleNamespace()
    sensing_start = "2017-02-12T01:12:30.0"
    fmt = "%Y-%m-%dT%H:%M:%S.%f"
    params.sensing_start = datetime.datetime.strptime(sensing_start, fmt)
    params.lon0 = 0.0
    params.lat0 = np.radians(lat)
    params.omega = np.radians(0.1)
    params.n_sv = 10
    params.n_samples = 20
    params.h_sat = 700000.0
    params.orbit_dt = 10.0
    params.epsg = 4326
    params.look_side = isce3.core.LookSide.Right
    params.ellipsoid = isce3.core.make_projection(params.epsg).ellipsoid
    params.orbit = compute_orbit(params)
    params.dem = "zero_dem.tif"
    return params


'''
def create_burst(ref_epoch_str="2017-02-12T01:12:30.0", az_time_interval):
    fmt = "%Y-%m-%dT%H:%M:%S.%f"
'''
def create_burst(params):

    # place holder value
    dont_matter_for_now = 0

    radar_freq = dont_matter_for_now
    wavelength = 0.24
    azimuth_steer_rate = 0.024389943375862838
    azimuth_time_interval = 2.0
    slant_range_time = dont_matter_for_now
    starting_range = 800000.0
    iw2_mid_range = dont_matter_for_now
    range_sampling_rate = dont_matter_for_now
    range_pxl_spacing = 10.0
    shape = (1, params.n_samples)
    az_fm_rate = dont_matter_for_now
    doppler = Doppler(isce3.core.Poly1d([1]), isce3.core.LUT2d())
    rng_processing_bandwidth = dont_matter_for_now
    pol = 'vv'
    burst_id = 't00_iw1_b000'
    platform_id = dont_matter_for_now
    center_pts = ()
    boundary_pts = []
    orbit_dir = str(dont_matter_for_now)
    tiff_path = dont_matter_for_now
    i_burst = dont_matter_for_now
    first_valid_sample = dont_matter_for_now
    last_sample = dont_matter_for_now
    first_valid_line = dont_matter_for_now
    last_line = dont_matter_for_now
    range_window_type = dont_matter_for_now
    range_window_coeff = dont_matter_for_now
    i_burst = dont_matter_for_now
    range_window_type = str(dont_matter_for_now)
    range_window_coeff = dont_matter_for_now
    rank = int(dont_matter_for_now)
    prf_raw_data = dont_matter_for_now

    burst = Sentinel1BurstSlc(params.sensing_start, radar_freq, wavelength,
                              azimuth_steer_rate, azimuth_time_interval,
                              slant_range_time, starting_range, iw2_mid_range,
                              range_sampling_rate, range_pxl_spacing,
                              shape, az_fm_rate, doppler,
                              rng_processing_bandwidth, pol, burst_id,
                              platform_id, center_pts,
                              boundary_pts, params.orbit, orbit_dir,
                              tiff_path, i_burst, first_valid_sample,
                              last_sample, first_valid_line, last_line,
                              range_window_type, range_window_coeff,
                              rank, prf_raw_data)

    return burst


def cfg_45_lat():
    cfg = SimpleNamespace()

    params = common_params(45)

    # create and set DEM raster
    zero_dem(params)

    # set rdr2geo params
    rdr2geo_params = SimpleNamespace()
    rdr2geo_params.threshold = 1e-8
    rdr2geo_params.numiter= 50
    rdr2geo_params.lines_per_block = 100
    rdr2geo_params.extraiter = 25
    rdr2geo_params.compute_mask = True
    cfg.rdr2geo_params = rdr2geo_params

    # make list with single burst
    cfg.bursts = [create_burst(params)]
    cfg.dem = params.dem
    cfg.gpu_enabled = False
    cfg.gpu_id = 0
    cfg.product_path = 'test_out'

    return cfg

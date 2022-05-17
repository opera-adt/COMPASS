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


def solve_geocentric_lat(slant_range, params):
    temp = 1 + params.h_sat / params.ellipsoid.a
    temp1 = slant_range / params.ellipsoid.a
    temp2 = slant_range / (params.ellipsoid.a + params.h_sat)

    cosang = 0.5 * (temp + (1.0/temp) - temp1 * temp2)
    angdiff = np.arccos(cosang);

    look_side_int = 1 if params.look_side == isce3.core.LookSide.Left \
        else -1
    if (look_side_int * params.omega > 0):
        x = params.lat0 + angdiff
    else:
        x = params.lat0 - angdiff

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
    params.slant_range0 = 80000.0
    params.d_slant_range = 10.0
    params.az_time0 = 5.0
    params.d_az_time = 2.0
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
    slant_range_time = dont_matter_for_now
    iw2_mid_range = dont_matter_for_now
    range_sampling_rate = dont_matter_for_now
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
                              azimuth_steer_rate, params.d_az_time,
                              slant_range_time, params.slant_range0, iw2_mid_range,
                              range_sampling_rate, params.d_slant_range,
                              shape, az_fm_rate, doppler,
                              rng_processing_bandwidth, pol, burst_id,
                              platform_id, center_pts,
                              boundary_pts, params.orbit, orbit_dir,
                              tiff_path, i_burst, first_valid_sample,
                              last_sample, first_valid_line, last_line,
                              range_window_type, range_window_coeff,
                              rank, prf_raw_data)

    return burst


def compute_expected_llh(test_params):
    i = np.arange(test_params.n_samples)
    ellipsoid = test_params.ellipsoid
    az_time = test_params.az_time0 + i * test_params.d_az_time
    slant_range = test_params.slant_range0 + i * test_params.d_slant_range
    lon = test_params.lon0 + test_params.omega * az_time
    c_lon = np.cos(lon)
    s_lon = np.sin(lon)
    lat = np.arccos(solve_geocentric_lat(slant_range, test_params))
    c_lat = np.cos(lat)
    s_lat = np.sin(lat)
    xyz_vec = np.vstack([ellipsoid.a * c_lat * c_lon,
                         ellipsoid.a * c_lat * s_lon,
                         ellipsoid.b * s_lat])
    llh = [ellipsoid.xyz_to_lon_lat(xyz_pt) for xyz_pt in xyz_vec]
    return llh


def cfg_45_lat():
    cfg = SimpleNamespace()

    cfg.test_params = common_params(45)

    # create and set DEM raster
    zero_dem(cfg.test_params)

    # set rdr2geo params
    rdr2geo_params = SimpleNamespace()
    rdr2geo_params.threshold = 1e-8
    rdr2geo_params.numiter= 50
    rdr2geo_params.lines_per_block = 100
    rdr2geo_params.extraiter = 25
    rdr2geo_params.compute_mask = True
    cfg.rdr2geo_params = rdr2geo_params

    # make list with single burst
    cfg.bursts = [create_burst(cfg.test_params)]
    cfg.dem = cfg.test_params.dem
    cfg.gpu_enabled = False
    cfg.gpu_id = 0
    cfg.product_path = 'test_out'

    return cfg

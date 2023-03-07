'''
Placeholder for model-based correction LUT
'''
import os
import isce3
from compass.utils.iono import get_ionex_value
from osgeo import gdal
import numpy as np
def cumulative_correction_luts(burst, dem_path, ionex_path,
                               rg_step=200, az_step=0.25,
                               scratch_path=None):
    '''
    Sum correction LUTs and returns cumulative correction LUT in slant range
    and azimuth directions

    Parameters
    ----------
    burst: Sentinel1BurstSlc
        Sentinel-1 A/B burst SLC object
    dem_path: str
        Path to the DEM file
    rg_step: float
        LUT spacing along slant range direction
    az_step: float
        LUT spacing along azimuth direction
    scratch_path: str
        Path to the scratch directory

    Returns
    -------
    rg_lut: isce3.core.LUT2d
        Sum of slant range correction LUTs in meters as a function of azimuth
        time and slant range
    az_lut: isce3.core.LUT2d
        Sum of azimuth correction LUTs in seconds as a function of azimuth time
        and slant range
    '''
    # Get individual LUTs
    geometrical_steer_doppler, bistatic_delay, az_fm_mismatch, ionosphere, dry_tropo = \
        compute_geocoding_correction_luts(burst,
                                          dem_path=dem_path,
                                          ionex_path=ionex_path,
                                          rg_step=rg_step,
                                          az_step=az_step,
                                          scratch_path=scratch_path)

    # Convert to geometrical doppler from range time (seconds) to range (m)
    rg_lut_data = \
        geometrical_steer_doppler.data * isce3.core.speed_of_light / 2.0 + dry_tropo + ionosphere

    # Invert signs to correct for convention
    az_lut_data = -(bistatic_delay.data + az_fm_mismatch.data)

    rg_lut = isce3.core.LUT2d(bistatic_delay.x_start,
                              bistatic_delay.y_start,
                              bistatic_delay.x_spacing,
                              bistatic_delay.y_spacing,
                              rg_lut_data)
    az_lut = isce3.core.LUT2d(bistatic_delay.x_start,
                              bistatic_delay.y_start,
                              bistatic_delay.x_spacing,
                              bistatic_delay.y_spacing,
                              az_lut_data)

    return rg_lut, az_lut


def compute_geocoding_correction_luts(burst, dem_path, ionex_path,
                                      rg_step=200, az_step=0.25,
                                      scratch_path=None):
    '''
    Compute slant range and azimuth LUTs corrections
    to be applied during burst geocoding
    Parameters
    ----------
    burst: Sentinel1BurstSlc
        S1-A/B burst object
    dem_path: str
        Path to the DEM required for azimuth FM rate mismatch.
    xstep: int
        LUT spacing along x/slant range in meters
    ystep: int
        LUT spacing along y/azimuth in seconds

    scratch_path: str
        Path to the scratch directory.
        If `None`, `burst.az_fm_rate_mismatch_mitigation()` will
        create temporary directory internally.

    Returns
    -------
    geometrical_steering_doppler: isce3.core.LUT2d:
        LUT2D object of total doppler (geometrical doppler +  steering doppler)
        in seconds as the function of the azimuth time and slant range.
        This correction needs to be added to the SLC tagged range time to
        get the corrected range times.

    bistatic_delay: isce3.core.LUT2d:
        LUT2D object of bistatic delay correction in seconds as a function
        of the azimuth time and slant range.
        This correction needs to be added to the SLC tagged azimuth time to
        get the corrected azimuth times.

    az_fm_mismatch: isce3.core.LUT2d:
        LUT2D object of azimuth FM rate mismatch mitigation,
        in seconds as the function of the azimuth time and slant range.
        This correction needs to be added to the SLC tagged azimuth time to
        get the corrected azimuth times.
    '''
    geometrical_steering_doppler = \
        burst.doppler_induced_range_shift(range_step=rg_step, az_step=az_step)

    bistatic_delay = burst.bistatic_delay(range_step=rg_step, az_step=az_step)

    if not os.path.exists(dem_path):
        raise FileNotFoundError(f'Cannot find the dem file: {dem_path}')

    az_fm_mismatch = burst.az_fm_rate_mismatch_mitigation(dem_path,
                                                          scratch_path,
                                                          range_step=rg_step,
                                                          az_step=az_step,
                                                          incidence_angle=True)

    lon_path = os.path.join(scratch_path, 'lon.rdr')
    lat_path = os.path.join(scratch_path, 'lat.rdr')
    hgt_path = os.path.join(scratch_path, 'hgt.rdr')
    inc_path = os.path.join(scratch_path, 'inc.rdr')

    ionosphere = ionosphere_delay(burst.sensing_mid, burst.wavelength,
                                  ionex_path, lon_path, lat_path, inc_path)
    
    dry_tropo = dry_tropo_delay(inc_path, hgt_path)

    return geometrical_steering_doppler, bistatic_delay, az_fm_mismatch, ionosphere, dry_tropo


def ionosphere_delay(sensing_time, wavelength,
                     ionex_path, lon_path, lat_path, inc_path):
    '''
    Calculate ionosphere delay for geolocation

    Parameters
    ----------
    time_sensing: datetime.datetime
        Sensing time of burst
    wavelength: float
        Wavelength of the signal
    lon_path: str
        Path to the longitude raster in radar grid
    lat_path: str
        Path to the latitude raster in radar grid
    inc_path: str
        Path to the incidence angle raster in radar grid
    
    Returns
    -------
    slant_range_delay: np.ndarray
        Ionospheric delay in slant range
    '''

    # Load the array
    arr_lon = gdal.Open(lon_path).ReadAsArray()
    arr_lat = gdal.Open(lat_path).ReadAsArray()
    arr_inc = gdal.Open(inc_path).ReadAsArray()

    if not ionex_path:
        raise RuntimeError('LUT correction was enabled, '
                           'but IONEX file was not provided in runconfig.')
    
    if not os.path.exists(ionex_path):
        raise RuntimeError(f'IONEX file was not found: {ionex_path}')

    utc_tod_sec = (sensing_time.hour * 3600.0 
                   + sensing_time.minute * 60.0
                   + sensing_time.second)

    ionex_val = get_ionex_value(ionex_path,
                                utc_tod_sec,
                                arr_lat.flatten(),
                                arr_lon.flatten())

    ionex_val = ionex_val.reshape(arr_lon.shape)

    freq_sensor = isce3.core.speed_of_light / wavelength
    electron_per_sqm = ionex_val * 1e16
    K = 40.31

    slant_range_delay = (K * electron_per_sqm / freq_sensor**2
                           / np.cos(np.deg2rad(arr_inc)))

    return slant_range_delay


def dry_tropo_delay(inc_path, hgt_path):
    '''
    Compute troposphere delay using static model

    Parameters:
    -----------
    inc_path: str
        Path to incidence angle raster in radar grid
    hgt_path: str
    Path to surface heightraster in radar grid

    Return:
    -------
    tropo: np.ndarray
        Troposphere delay in slant range
    '''
    ZPD = 2.3
    H = 6000.0
    arr_inc = gdal.Open(inc_path).ReadAsArray()
    arr_hgt = gdal.Open(hgt_path).ReadAsArray()

    tropo = ZPD / np.cos(np.deg2rad(arr_inc)) * np.exp(-1 * arr_hgt / H)
    return tropo

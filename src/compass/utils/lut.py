'''
Placeholder for model-based correction LUT
'''
import os
import isce3
from compass.utils.iono import get_ionex_value
from osgeo import gdal
import numpy as np
def cumulative_correction_luts(burst, dem_path,
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
    geometrical_steer_doppler, bistatic_delay, az_fm_mismatch = \
        compute_geocoding_correction_luts(burst,
                                          dem_path=dem_path,
                                          rg_step=rg_step,
                                          az_step=az_step,
                                          scratch_path=scratch_path)

    # Convert to geometrical doppler from range time (seconds) to range (m)
    rg_lut_data = \
        geometrical_steer_doppler.data * isce3.core.speed_of_light / 2.0

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


def compute_geocoding_correction_luts(burst, dem_path,
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

    return geometrical_steering_doppler, bistatic_delay, az_fm_mismatch


def ionosphere(burst,
               path_lon, path_lat, path_inc, path_ionex,
               scratch_path=None):
    '''
    Calculate ionosphere delay for geolocation 
    '''

    # Load the array
    arr_lon = gdal.Open(path_lon).ReadAsArray()
    arr_lat = gdal.Open(path_lat).ReadAsArray()
    arr_inc = gdal.Open(path_inc).ReadAsArray()

    utc_tod_sec = (burst.sensing_mid.hour * 3600.0 
                   + burst.sensing_mid.minute * 60.0
                   + burst.sensing_mid.second)
    

    ionex_val = get_ionex_value(path_ionex,
                                utc_tod_sec,
                                arr_lat.flatten(),
                                arr_lon.flatten())

    ionex_val = ionex_val.reshape(arr_lon.shape)

    freq_sensor = isce3.core.speed_of_light / burst.wavelength
    electron_per_sqm = ionex_val * 1e16
    K=40.31

    slant_range_delay = K * electron_per_sqm / freq_sensor**2 / np.cos(np.deg2rad(arr_inc))

    return slant_range_delay


# test code. Remove before commit
if __name__=='__main__':
    '''
    # test code. Remove before commit
    '''
    import s1reader

    os.chdir('/Users/jeong/Documents/OPERA_SCRATCH/CSLC/IONOSPHERE_TEST_SITE')
    filename_lon = '/Users/jeong/Documents/OPERA_SCRATCH/CSLC/SET_TEST/scratch_s1_cslc_set_on/t064_135523_iw2/20221016/lon_20230304_112753451171.rdr'
    filename_lat = '/Users/jeong/Documents/OPERA_SCRATCH/CSLC/SET_TEST/scratch_s1_cslc_set_on/t064_135523_iw2/20221016/lat_20230304_112753451171.rdr'
    filename_inc = '/Users/jeong/Documents/OPERA_SCRATCH/CSLC/SET_TEST/scratch_s1_cslc_set_on/t064_135523_iw2/20221016/inc_20230304_112753451171.rdr'

    filename_ionex = '/Users/jeong/Documents/OPERA_SCRATCH/CSLC/IONOSPHERE_TEST_SITE/jplg2890.22i'

    path_safe = '/Users/jeong/Documents/OPERA_SCRATCH/CSLC/IONOSPHERE_TEST_SITE/input/S1A_IW_SLC__1SDV_20221016T015043_20221016T015111_045461_056FC0_6681.zip'
    path_orbit = '/Users/jeong/Documents/OPERA_SCRATCH/CSLC/IONOSPHERE_TEST_SITE/input/S1A_OPER_AUX_POEORB_OPOD_20221105T083813_V20221015T225942_20221017T005942.EOF'

    bursts = s1reader.load_bursts(path_safe, path_orbit, 2, 'VV')
    burst_cr = bursts[5]

    ionosphere(burst_cr, filename_lon, filename_lat, filename_inc, filename_ionex)
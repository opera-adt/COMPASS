'''
Placeholder for model-based correction LUT
'''
import os
import isce3
from compass.utils.helpers import write_raster


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
    geometry_doppler = geometrical_steer_doppler.data * isce3.core.speed_of_light / 2.0
    rg_lut_data = geometry_doppler

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

    # Save corrections on disk. In this way, we should avoid running
    # the corrections again when allocating data inside the HDF5 product
    # Create a directory in the scratch path to save corrections
    output_path = f'{scratch_path}/corrections'
    os.makedirs(output_path, exist_ok=True)
    data_list = [geometry_doppler, bistatic_delay.data, az_fm_mismatch.data]
    descr = ['geometrical doppler', 'bistatic delay', 'azimuth FM rate mismatch']

    write_raster(f'{output_path}/corrections', data_list, descr)

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
                                                          az_step=az_step)

    return geometrical_steering_doppler, bistatic_delay, az_fm_mismatch

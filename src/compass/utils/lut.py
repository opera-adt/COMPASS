def compute_geocoding_correction_luts(burst, rg_step=200, az_step=0.25):
    '''
    Compute slant range and azimuth LUTs corrections
    to be applied during burst geocoding

    Parameters
    ----------
    burst: Sentinel1BurstSlc
        S1-A/B burst object
    xstep: int
        LUT spacing (in pixels) along x/slant range in meters
    ystep: int
        LUT spacing (in pixels) along x/azimuth in seconds

    Returns
    -------
    rg_lut: isce3.core.LUT2d:
        2D array containing sum of range corrections
        LUT2D object of bistatic delay correction in seconds as a function
        of the azimuth time and slant range, or range and azimuth indices.
        This correction needs to be added to the SLC tagged azimuth time to
        get the corrected azimuth times.
    az_lut: isce3.core.LUT2d:
        LUT2D object of range delay correction [seconds] as a function
        of the azimuth time and slant range, or x and y indices.
    '''

    bistatic_delay = burst.bistatic_delay(range_step=rg_step, az_step=az_step)
    geometrical_steering_doppler= burst.geometrical_and_steering_doppler(range_step=rg_step,
                                                                         az_step=az_step)

    return geometrical_steering_doppler, bistatic_delay

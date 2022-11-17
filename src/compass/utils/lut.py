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
    rg_lut: ndarray
        2D array containing sum of range corrections
    az_lut: ndarray
        2D array containg sum of azimuth corrections
    '''

    az_bistatic = burst.bistatic_delay(range_stepx=rg_step, az_step=az_step)
    _, _, rg_doppler = burst.geometrical_and_steering_doppler(range_step=rg_step, az_step=az_step)

    rg_lut = rg_doppler
    az_lut = az_bistatic.data

    return rg_lut, az_lut
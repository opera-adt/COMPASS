def compute_lut(burst, xstep=500, ystep=50):
    '''
    Compute slant range and azimuth LUTs corrections
    to be applied during burst geocoding

    Parameters
    ----------
    burst: Sentinel1BurstSlc
        S1-A/B burst object
    xstep: int
        LUT spacing (in pixels) along x/slant range
    ystep: int
        LUT spacing (in pixels) along x/azimuth

    Returns
    -------
    rg_lut: ndarray
        2D array containing sum of range corrections
    az_lut: ndarray
        2D array containg sum of azimuth corrections
    '''

    az_bistatic = burst.bistatic_delay(xstep=xstep, ystep=ystep)
    _, _, rg_doppler = burst.geometrical_and_steering_doppler(xstep=xstep, ystep=ystep)

    rg_lut = rg_doppler
    az_lut = az_bistatic.data

    return rg_lut, az_lut
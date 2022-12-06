def compute_geocoding_correction_luts(burst, rg_step=200, az_step=0.25, path_dem=None, path_scratch=None):
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
    path_dem: str
        Path to the DEM required for azimuth FM rate mismatch.
        If `None`, the calculation of the FM mismatch will be skipped.
    path_scratch: str
        Path to the scratch directory.
        If `None`, `burst.az_fm_rate_mismatch_mitigation()` will
        create temporary directory internally.

    Returns
    -------
    rg_lut: ndarray
        2D array containing sum of range corrections
    az_lut: ndarray
        2D array containg sum of azimuth corrections
    '''

    az_bistatic = burst.bistatic_delay(range_step=rg_step, az_step=az_step)
    rg_doppler = burst.geometrical_and_steering_doppler(range_step=rg_step, az_step=az_step)

    if path_dem is None:
        # Skip the FM mismatch rate calculation
        az_fm_mismatch = None
    else:
        az_fm_mismatch = burst.az_fm_rate_mismatch_mitigation(path_dem,
                                                              path_scratch,
                                                              rg_step=rg_step,
                                                              az_step=az_step)

    rg_lut = rg_doppler
    az_lut = az_bistatic.data

    if not az_fm_mismatch is None:
        az_lut += az_fm_mismatch.data

    return rg_lut, az_lut

############################################################################
# Suite of geometry utilities for SAR processing                           #
# This was adapted from MintPy on 02-22-2023                               #
# Source code is available at:                                             #
# https://github.com/insarlab/MintPy/blob/main/src/mintpy/utils/utils0.py  #
############################################################################


import numpy as np
import isce3

def los2orbit_azimuth_angle(los_az_angle, look_direction='right'):
    """
    Convert the azimuth angle of the LOS vector to the one of the orbit flight vector.
    The conversion done for this function only works for zero-Doppler geometry.

    Parameters
    ----------
    los_az_angle: np.ndarray or float
        Azimuth angle of the LOS vector from the ground to the SAR platform measured from
        the north with anti-clockwise direction as positive, in the unit of degrees

    Returns
    -------
    orb_az_angle: np.ndarray or float
        Azimuth angle of the SAR platform along track/orbit direction measured from
        the north with anti-clockwise direction as positive, in the unit of degrees
    """

    if look_direction == 'right':
        orb_az_angle = los_az_angle - 90
    else:
        orb_az_angle = los_az_angle + 90
    orb_az_angle -= np.round(orb_az_angle / 360.) * 360.
    return orb_az_angle


def azimuth2heading_angle(az_angle, look_direction='right'):
    """
    Convert azimuth angle from ISCE los.rdr band2 into satellite orbit heading angle
    ISCE-2 los.* file band2 is azimuth angle of LOS vector from ground target to the satellite
    measured from the north in anti-clockwise as positive.

    Below are typical values in deg for satellites with near-polar orbit:
        ascending  orbit: heading angle of -12  and azimuth angle of 102
        descending orbit: heading angle of -168 and azimuth angle of -102

    Parameters
    ----------
    az_angle: np.ndarray or float
        Measured from North in anti-clockwise direction. Same definition
        as ISCE2 azimmuth angle (second band of *los raster)
    look_direction: str
        Satellite look direction. S1-A/B is right; NISAR is left

    Returns
    -------
    head_angle: np.ndarray or float
        Azimuth angle from ground target to the satellite measured
        from the North in anti-clockwise direction as positive
    """

    if look_direction == 'right':
        head_angle = (az_angle - 90) * -1
    else:
        head_angle = (az_angle + 90) * -1
    head_angle -= np.round(head_angle / 360.) * 360.
    return head_angle


def heading2azimuth_angle(head_angle, look_direction='right'):
    """
    Convert satellite orbit heading angle into azimuth angle as defined in ISCE-2

    Parameters
    ----------
    head_angle: np.ndarray or float
        Azimuth angle from ground target to the satellite measured
        from the North in anti-clockwise direction as positive
    look_direction: str
        Satellite look direction. S1-A/B is right; NISAR is left

    Returns
    -------
    az_angle: np.ndarray or float
        Measured from the North in anti-clockwise direction. Same definition
        as ISCE2 azimuth angle (second band of *los raster)
    """
    if look_direction == 'right':
        az_angle = (head_angle - 90) * -1
    else:
        az_angle = (head_angle + 90) * -1
    az_angle -= np.round(az_angle / 360.) * 360.
    return az_angle


def enu2los(v_e, v_n, v_u, inc_angle, head_angle=None, az_angle=None):
    """
    Project East/North/Up motion into the line-of-sight (LOS)
    direction defined by incidence/azimuth angle.

    Parameters
    ----------
    v_e: np.ndarray or float
        displacement in East-West direction, East  as positive
    v_n: np.ndarray or float
        displacement in North-South direction, North as positive
    v_u: np.ndarray or float
        displacement in vertical direction, Up as positive
    inc_angle: np.ndarray or float
        incidence angle from vertical, in the unit of degrees
    head_angle: np.ndarray or float
        azimuth angle of the SAR platform along track direction measured from
        the North with clockwise direction as positive, in the unit of degrees
    az_angle: np.ndarray or float
        azimuth angle of the LOS vector from the ground to the SAR platform
        measured from the north with anti-clockwise direction as positive, in the unit of degrees
        head_angle = 90 - az_angle

    Returns
    -------
    v_los: np.ndarray or float
        displacement in LOS direction, motion toward satellite as positive
    """

    if az_angle is None:
        if head_angle is not None:
            az_angle = heading2azimuth_angle(head_angle)
        else:
            raise ValueError(f'invalid az_angle: {az_angle}!')

    # project ENU onto LOS
    v_los = (  v_e * np.sin(np.deg2rad(inc_angle)) * np.sin(np.deg2rad(az_angle)) * -1
             + v_n * np.sin(np.deg2rad(inc_angle)) * np.cos(np.deg2rad(az_angle))
             + v_u * np.cos(np.deg2rad(inc_angle)))

    return v_los


def en2az(v_e, v_n, orb_az_angle):
    """
    Project east/north motion into the radar azimuth direction.
    Parameters
    ----------
    v_e: np.ndarray or float
        displacement in East-West   direction, East  as positive
    v_n: np.ndarray or float
        displacement in North-South direction, North as positive
    orb_az_angle: np.ndarray or float
        azimuth angle of the SAR platform along track/orbit direction
        measured from the north with anti-clockwise direction as positive, in the unit of degrees
        orb_az_angle = los_az_angle + 90 for right-looking radar.

    Returns
    -------
    v_az: np.ndarray or float
        displacement in azimuth direction,
        motion along flight direction as positive
    """
    # project EN onto azimuth
    v_az = (  v_e * np.sin(np.deg2rad(orb_az_angle)) * -1
            + v_n * np.cos(np.deg2rad(orb_az_angle)))
    return v_az


def calc_azimuth_from_east_north_obs(east, north):
    """
    Calculate the azimuth angle of the given horizontal observation (in East and North)

    Parameters
    ----------
    east: float
        eastward motion
    north: float
        northward motion

    Returns
    -------
    az_angle: float
        azimuth angle in degree measured from the north
        with anti-clockwise as positive
    """

    az_angle = -1 * np.rad2deg(np.arctan2(east, north)) % 360
    return az_angle


def get_unit_vector4component_of_interest(los_inc_angle, los_az_angle, comp='enu2los', horz_az_angle=None):
    """
    Get the unit vector for the component of interest.

    Parameters
    ----------
    los_inc_angle: np.ndarray or float
        incidence angle from vertical, in the unit of degrees
    los_az_angle: np.ndarray or float
        azimuth angle of the LOS vector from the ground to the SAR platform
        measured from the north with anti-clockwise direction as positive, in the unit of degrees
    comp: str
        component of interest. It can be one of the following values
        enu2los, en2los, hz2los, u2los, up2los, orb(it)_az, vert, horz
    horz_az_angle: np.ndarray or float
        azimuth angle of the horizontal direction of interest measured from
         the north with anti-clockwise direction as positive, in the unit of degrees

    Returns
    -------
    unit_vec: list(np.ndarray/float)
        unit vector of the ENU component for the component of interest
    """

    # check input arguments
    comps = [
        'enu2los', 'en2los', 'hz2los', 'horz2los', 'u2los', 'vert2los',   # radar LOS / cross-track
        'en2az', 'hz2az', 'orb_az', 'orbit_az',                           # radar azimuth / along-track
        'vert', 'vertical', 'horz', 'horizontal',                         # vertical / horizontal
    ]

    if comp not in comps:
        raise ValueError(f'un-recognized comp input: {comp}.\nchoose from: {comps}')

    if comp == 'horz' and horz_az_angle is None:
        raise ValueError('comp=horz requires horz_az_angle input!')

    # initiate output
    unit_vec = None

    if comp in ['enu2los']:
        unit_vec = [
            np.sin(np.deg2rad(los_inc_angle)) * np.sin(np.deg2rad(los_az_angle)) * -1,
            np.sin(np.deg2rad(los_inc_angle)) * np.cos(np.deg2rad(los_az_angle)),
            np.cos(np.deg2rad(los_inc_angle)),
        ]

    elif comp in ['en2los', 'hz2los', 'horz2los']:
        unit_vec = [
            np.sin(np.deg2rad(los_inc_angle)) * np.sin(np.deg2rad(los_az_angle)) * -1,
            np.sin(np.deg2rad(los_inc_angle)) * np.cos(np.deg2rad(los_az_angle)),
            np.zeros_like(los_inc_angle),
        ]

    elif comp in ['u2los', 'vert2los']:
        unit_vec = [
            np.zeros_like(los_inc_angle),
            np.zeros_like(los_inc_angle),
            np.cos(np.deg2rad(los_inc_angle)),
        ]

    elif comp in ['en2az', 'hz2az', 'orb_az', 'orbit_az']:
        orb_az_angle = los2orbit_azimuth_angle(los_az_angle)
        unit_vec = [
            np.sin(np.deg2rad(orb_az_angle)) * -1,
            np.cos(np.deg2rad(orb_az_angle)),
            np.zeros_like(orb_az_angle),
        ]

    elif comp in ['vert', 'vertical']:
        unit_vec = [0, 0, 1]

    elif comp in ['horz', 'horizontal']:
        unit_vec = [
            np.sin(np.deg2rad(horz_az_angle)) * -1,
            np.cos(np.deg2rad(horz_az_angle)),
            np.zeros_like(horz_az_angle),
        ]

    return unit_vec


def enu2rgaz(radargrid_ref, orbit, ellipsoid,
             lon_arr, lat_arr, hgt_arr,
             e_arr, n_arr, u_arr,
             geo2rdr_params=None):
    '''
    Convert ENU displacement into range / azimuth displacement,
    based on the idea mentioned in ETAD ATBD, available in the link below:
    ("https://sentinels.copernicus.eu/documents/247904/4629150/"
     "ETAD-DLR-DD-0008_Algorithm-Technical-Baseline-Document_2.3.pdf/"
     "5cb45b43-76dc-8dec-04ef-ca1252ace434?t=1680181574715")

    Algorithm description
    ---------------------
    For all lon / lat / height of the array;
    1. Calculate the ECEF coordinates before applying SET
    2. Calculate the normal vector of east / north / up directions of the point
    3. Scale the normal vector in 2 with ENU displacement to
       get the displacement in ECEF
    4. Add the vectors calculated in 4 into 1.
       This will be the ECEF coordinates after applying SET
    5. Convert 4 into lat / lon / hgt.
       This will be LLH coordinates after applying SET
    6. Calculate the radar coordinate before SET applied using `geo2rdr`
    7. Calculate the radar coordinate AFTER SET applied using `geo2rdr`
    8. Calculate the difference between (7) and (6),
       which will be the displacement in radargrid by SET

    Parameters
    ----------
    radargrid_ref: isce3.product.RadarGridParameters
        Radargrid of the burst
    orbit: isce3.core.Orbit
        Orbit of the burst
    ellipsoid: isce3.core.Ellipsoid
        Ellipsoid definition
    lon_arr, lat_arr, hgt_arr: np.nadrray
        Arrays for longitude, latitude, and height.
        Units for longitude and latitude are degree; unit for height is meters.
    e_arr, n_arr, u_arr: np.ndarray
        Displacement in east, north, and up direction in meters
    geo2rdr_params: SimpleNameSpace
        Parameters for geo2rdr

    Returns
    -------
    rg_arr: np.ndarray
        Displacement in slant range direction in meters.
    az_arr: np.ndarray
        Displacement in azimuth direction in seconds.

    Notes
    -----
    When `geo2rdr_params` is not provided, then the iteration
    threshold and max # iterations are set to
    `1.0e-8` and `25` respectively.

    '''
    if geo2rdr_params is None:
        # default threshold and # iteration for geo2rdr
        threshold = 1.0e-8
        maxiter = 25
    else:
        threshold = geo2rdr_params.threshold
        maxiter = geo2rdr_params.numiter

    shape_arr = lon_arr.shape
    rg_arr = np.zeros(shape_arr)
    az_arr = np.zeros(shape_arr)

    # Calculate the ENU vector in ECEF
    for i, lon_deg in enumerate(np.nditer(lon_arr)):
        index_arr = np.unravel_index(i, lon_arr.shape)
        lat_deg = lat_arr[index_arr]
        hgt = hgt_arr[index_arr]

        vec_e, vec_n, vec_u = get_enu_vector_ecef(lon_deg, lat_deg)

        llh_ref = np.array([np.deg2rad(lon_deg),
                            np.deg2rad(lat_deg),
                            hgt])

        xyz_before = ellipsoid.lon_lat_to_xyz(llh_ref)
        xyz_after_set = (xyz_before
                         + vec_e * e_arr[index_arr]
                         + vec_n * n_arr[index_arr]
                         + vec_u * u_arr[index_arr])
        llh_displaced = ellipsoid.xyz_to_lon_lat(xyz_after_set)

        aztime_ref, slant_range_ref =\
            isce3.geometry.geo2rdr(llh_ref,
                                   ellipsoid,
                                   orbit,
                                   isce3.core.LUT2d(),
                                   radargrid_ref.wavelength,
                                   radargrid_ref.lookside,
                                   threshold=threshold,
                                   maxiter=maxiter)

        aztime_displaced, slant_range_displaced =\
            isce3.geometry.geo2rdr(llh_displaced,
                                   ellipsoid,
                                   orbit,
                                   isce3.core.LUT2d(),
                                   radargrid_ref.wavelength,
                                   radargrid_ref.lookside,
                                   threshold=threshold,
                                   maxiter=maxiter)

        rg_arr[index_arr] = slant_range_displaced - slant_range_ref
        az_arr[index_arr] = aztime_displaced - aztime_ref

    return rg_arr, az_arr


def get_enu_vector_ecef(lon, lat, units='degrees'):
    '''
    Calculate the east, north, and up vectors in ECEF for lon / lat provided

    Parameters
    ----------
    lon: np.ndarray
        Longitude of the points to calculate ENU vectors
    lat: np.ndarray
        Latitude of the points to calculate ENU vectors
    units: str
        Units of the `lon` and `lat`.
        Acceptable units are `radians` or `degrees`, (Default: degrees)

    Returns
    -------
    vec_e: np.ndarray
        unit vector of "east" direction in ECEF
    vec_n: np.ndarray
        unit vector of "north" direction in ECEF
    vec_u: np.ndarray
        unit vector of "up" direction in ECEF
    '''
    if units == 'degrees':
        lon_rad = np.deg2rad(lon)
        lat_rad = np.deg2rad(lat)
    elif units == 'radians':
        lon_rad = lon
        lat_rad = lat
    else:
        raise ValueError(f'"{units}" was provided for `units`, '
                         'which needs to be either `degrees` or `radians`')

    vec_u = np.array([np.cos(lon_rad) * np.cos(lat_rad),
                      np.sin(lon_rad) * np.cos(lat_rad),
                      np.sin(lat_rad)])

    vec_n = np.array([-np.cos(lon_rad) * np.sin(lat_rad),
                      -np.sin(lon_rad) * np.sin(lat_rad),
                      np.cos(lat_rad)])

    vec_e = np.cross(vec_n, vec_u, axis=0)

    return vec_e, vec_n, vec_u

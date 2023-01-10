'''
Placeholder for model-based correction LUT
'''


import os
import isce3
import pysolid
import numpy as np
from osgeo import gdal
from compass.utils.geo_grid import transform_coordinates
from scipy.interpolate import RegularGridInterpolator as RGI
from skimage.transform import resize


def compute_geocoding_correction_luts(burst, geogrid, dem_path,
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
        in seconds as the function of the azimuth time and slant range,
        or range and azimuth indices.
        This correction needs to be added to the SLC tagged azimuth time to
        get the corrected azimuth times.

    bistatic_delay: isce3.core.LUT2d:
        LUT2D object of bistatic delay correction in seconds as a function
        of the azimuth time and slant range, or range and azimuth indices.
        This correction needs to be added to the SLC tagged range time to
        get the corrected range times.

    az_fm_mismatch: isce3.core.LUT2d:
        LUT2D object of azimuth FM rate mismatch mitigation,
        in seconds as the function of the azimuth time and slant range,
        or range and azimuth indices.
        This correction needs to be added to the SLC tagged azimuth time to
        get the corrected azimuth times.
    '''

    bistatic_delay = burst.bistatic_delay(range_step=rg_step, az_step=az_step)
    geometrical_steering_doppler= burst.geometrical_and_steering_doppler(range_step=rg_step,
                                                                         az_step=az_step)
    if not os.path.exists(dem_path):
        raise FileNotFoundError(f'Cannot find the dem file: {dem_path}')

    az_fm_mismatch = burst.az_fm_rate_mismatch_mitigation(dem_path,
                                                          scratch_path,
                                                          range_step=rg_step,
                                                          az_step=az_step)

    # Compute Solid Earth Tides
    rg_set, az_set = solid_earth_tides(burst, geogrid, dem_path,
                                       scratch_path)

    # Resize SET to the size of the correction grid
    out_shape = bistatic_delay.data.shape
    kwargs = dict(order=1, mode='edge', anti_aliasing=True,
                  preserve_range=True)
    new_rg_set = resize(rg_set, out_shape, **kwargs)
    new_az_set = resize(az_set, out_shape, **kwargs)

    return geometrical_steering_doppler, bistatic_delay, az_fm_mismatch, [new_rg_set, new_az_set]


def solid_earth_tides(burst, geogrid, dem_path, scratchdir):
    '''
    Compute displacement due to Solid Earth Tides (SET)
    in slant range and azimuth directions
    Parameters
    ---------
    burst: Sentinel1Slc
        S1-A/B burst object
    geogrid: isce3.product.geogrid
        Geogrid of the output CSLC product
    dem_path: str
        File path to available DEM
    scratchdir: str
        Path to scratch directory
    Returns
    ------
    rg_set: np.ndarray
        2D array with SET displacement along LOS
    az_set: np.ndarray
        2D array with SET displacement along azimuth
    '''

    # Get ellipsoid
    dem_raster = isce3.io.Raster(dem_path)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # Create directory where to allocate SET results
    output_path = f'{scratchdir}/solid_earth_tides'
    os.makedirs(output_path, exist_ok=True)

    # Compute SET on coarse geogrid (~ 2.5 km x 2.5 km)
    # Note: SET computation is only in lat/lon
    lat_start, lon_start = transform_coordinates(geogrid.start_x,
                                                 geogrid.start_y,
                                                 geogrid.epsg,
                                                 4326)

    # Prepare atr object for SET computation
    atr = {
        'LENGTH': int(geogrid.length / 250.0),
        'WIDTH': int(geogrid.width / 500.0),
        'X_FIRST': lon_start,
        'Y_FIRST': lat_start,
        'X_STEP': 0.023,
        'Y_STEP': 0.023}

    # Compute East, North, UP SET displacement with pySolid
    # at the time of the burst sensing start
    (tide_e,
     tide_n,
     tide_u) = pysolid.calc_solid_earth_tides_grid(burst.sensing_start, atr,
                                                   display=False, verbose=True)

    # Compute topo layers to convert SET from ENU to radar coordinates
    compute_rdr2geo_rasters(burst, ellipsoid, dem_raster, output_path)

    # Resample the SET components to the radar grid
    # Create lat/lon arrays for the SET geogrid
    lat_geo_array = np.linspace(lat_start,
                                lat_start + atr['Y_STEP'] * atr['LENGTH'],
                                num=atr['LENGTH'])
    lon_geo_array = np.linspace(lon_start,
                                lon_start + atr['X_STEP'] * atr['WIDTH'],
                                num=atr['WIDTH'])
    # Get lat/lon array for radar grid computed with rdr2geo
    # Get the lat/lon radar coordinates grids (from rdr2geo)
    lat_radar_grid = open_raster(f'{output_path}/y.rdr')
    lon_radar_grid = open_raster(f'{output_path}/x.rdr')

    # Create array of source and destination points
    # Note: flip lats to be consistent with RGI
    pts_src = (np.flipud(lat_geo_array), lon_geo_array)
    pts_dest = (lat_radar_grid.flatten(), lon_radar_grid.flatten())

    # Use scipy to resample tides from geo to radar grid
    rdr_tide_e = resample_tide(tide_e, pts_src, pts_dest).reshape(
        lat_radar_grid.shape)
    rdr_tide_n = resample_tide(tide_n, pts_src, pts_dest).reshape(
        lat_radar_grid.shape)
    rdr_tide_u = resample_tide(tide_u, pts_src, pts_dest).reshape(
        lat_radar_grid.shape)

    # Project SET from ENU to range/azimuth direction
    inc_angle = open_raster(f'{output_path}/incidence_angle.rdr')
    head_angle = open_raster(f'{output_path}/heading_angle.rdr')
    set_rg = enu2los(rdr_tide_e, rdr_tide_n, rdr_tide_u, inc_angle, head_angle=head_angle)
    set_az = en2az(rdr_tide_e, rdr_tide_n, head_angle)

    return set_rg, set_az


def compute_rdr2geo_rasters(burst, ellipsoid, dem_raster, output_path):
    '''
    Get latitude, longitude, incidence and
    azimuth angle on multi-looked radar grid
    Parameters
    ----------
    burst: Sentinel1Slc
        S1-A/B burst object
    ellipsoid: isce3.ext.isce3.core
        ISCE3 Ellipsoid object
    dem_raster: isce3.io.Raster
        ISCE3 object including DEM raster
    output_path: str
        Path where to save output rasters
    '''

    # Get radar and doppler grid
    rdr_grid = burst.as_isce3_radargrid()
    coarse_rdr_grid = rdr_grid.multilook(64, 429)
    grid_doppler = isce3.core.LUT2d()

    # Initialize the rdr2geo object
    rdr2geo_obj = isce3.geometry.Rdr2Geo(coarse_rdr_grid, burst.orbit,
                                         ellipsoid, grid_doppler,
                                         threshold=1.0e8)

    # Get the rdr2geo raster needed for SET computation
    topo_output = {'x': (True, gdal.GDT_Float64),
                   'y': (True, gdal.GDT_Float64),
                   'incidence_angle': (True, gdal.GDT_Float32),
                   'heading_angle': (True, gdal.GDT_Float32)}
    raster_list = [
        isce3.io.Raster(f'{output_path}/{fname}.rdr', coarse_rdr_grid.width,
                        coarse_rdr_grid.length, 1, dtype, 'ENVI')
        if enabled else None
        for fname, (enabled, dtype) in topo_output.items()]
    x_raster, y_raster, incidence_raster, heading_raster = raster_list

    # Run rdr2geo on coarse radar grid
    rdr2geo_obj.topo(dem_raster, x_raster, y_raster,
                     incidence_angle_raster=incidence_raster,
                     heading_angle_raster=heading_raster)


def open_raster(filename, band=1):
    '''
    Return band as numpy array from gdal-friendly raster
    Parameters
    ----------
    filename: str
        Path where is stored GDAL raster to open
    band: int
        Band number to open
    Returns
    -------
    raster: np.ndarray
        Numpy array containing the raster band to open
    '''

    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    raster = ds.GetRasterBand(band).ReadAsArray()
    return raster


def resample_tide(geo_tide, pts_src, pts_dest):
    '''
    Use scipy RegularGridInterpolator to resample geo_tide
    from a geographical to a radar grid
    Parameters
    ----------
    geo_tide: np.ndarray
        Tide displacement component on geographical grid
    pts_src: tuple of ndarray
        Points defining the source rectangular regular grid for resampling
    pts_dest: tuple of ndarray
        Points defining the destination grid for resampling
    Returns
    -------
    rdr_tide: np.ndarray
        Tide displacement component resampled on radar grid
    '''
    # Flip tide displacement component to be consistent with flipped latitudes
    geo_tide = np.flipud(geo_tide)
    rgi_func = RGI(pts_src, geo_tide, method='nearest',
                   bounds_error=False, fill_value=0)
    rdr_tide = rgi_func(pts_dest)
    return rdr_tide


def heading2azimuth_angle(head_angle, look_direction='right'):
    '''
    Convert satellite orbit heading angle into azimuth angle
    Parameters
    ----------
    head_angle: np.ndarray
        azimuth angle of the SAR platform along track direction
        measured from the East with anti-clockwise direction as positive (degrees)
    look_direction: str
        Sensor look-direction (left or right)
    Returns
    -------
    az_angle: np.ndarray
        azimuth angle of the LOS vector from the ground to the SAR platform
        measured from the north with anti-clockwise direction as positive (degrees)
    '''
    if look_direction == 'right':
        az_angle = (head_angle + 180)
    else:
        az_angle = (head_angle - 180)
    az_angle -= np.round(az_angle / 360.) * 360.
    return az_angle


def enu2los(x_e, x_n, x_u, inc_angle, head_angle=None, az_angle=None):
    '''
    Project east/north/up motion into the line-of-sight (LOS)
    direction defined by incidence/azimuth angle
    Parameters
    ----------
    x_e: np.ndarray or float
        Displacement in east-west direction, east  as positive
    x_n: np.ndarray or float
        Displacement in north-south direction, north as positive
    x_u: np.ndarray or float
        Displacement in vertical direction, up as positive
    inc_angle: np.ndarray or float
        Incidence angle from vertical, degrees
    head_angle: np.ndarray or float
        Azimuth angle of the SAR platform along track direction
        measured from East with anti-clockwise direction as positive, degrees
    az_angle: np.ndarray or float
        Azimuth angle of the LOS vector from the ground to the SAR platform
        measured from the north with anti-clockwise direction as positive,  degrees
    Returns
    -------
    x_los: np.ndarray or float
        Displacement in LOS direction, motion toward satellite as positive
    '''

    if az_angle is None:
        if head_angle is not None:
            az_angle = heading2azimuth_angle(head_angle)
        else:
            raise ValueError(f'invalid az_angle: {az_angle}!')

    # project ENU onto LOS
    x_los = (  x_e * np.sin(np.deg2rad(inc_angle)) * np.sin(np.deg2rad(az_angle)) * -1
             + x_n * np.sin(np.deg2rad(inc_angle)) * np.cos(np.deg2rad(az_angle))
             + x_u * np.cos(np.deg2rad(inc_angle)))

    return x_los


def en2az(x_e, x_n, head_angle):
    '''
    Project east/north motion into the radar azimuth direction
    Parameters
    ----------
    x_e: np.ndarray or float
        Displacement in east-west direction, east  as positive
    x_n: np.ndarray or float
        Displacement in north-south direction, north as positive
    head_angle: np.ndarray or float
        Azimuth angle of the SAR platform along track direction
        measured from East with anti-clockwise direction as positive, degrees
    Returns
    -------
    x_az: np.ndarray or float
        displacement in azimuth direction, motion along flight
        direction as positive
    '''

    # Note to compute x_az, we need the orbit azimuth angle defined as
    # the azimuth angle of the SAR platform along track/orbit direction
    # measured from the north with anti-clockwise direction as positive
    # For right-looking radar this should be:
    orb_az_angle = head_angle + 45

    # project EN onto azimuth
    x_az = (  x_e * np.sin(np.deg2rad(orb_az_angle)) * -1
            + x_n * np.cos(np.deg2rad(orb_az_angle)))
    return x_az

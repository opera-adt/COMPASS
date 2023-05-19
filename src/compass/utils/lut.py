'''
Placeholder for model-based correction LUT
'''
import os
import isce3
import numpy as np
from osgeo import gdal
import pysolid
from scipy.interpolate import RegularGridInterpolator as RGI
from skimage.transform import resize

from compass.utils.geometry_utils import enu2los, en2az
from compass.utils.h5_helpers import Meta, add_dataset_and_attrs
from compass.utils.helpers import open_raster
from compass.utils.iono import ionosphere_delay


def correction_luts(burst, lut_par, dem_path, tec_path, h5_file_obj,
                    scratch_path=None,
                    weather_model_path=None):
    '''
    Compute correction look-up tables (LUTs)

    Parameters
    ----------
    burst: Sentinel1BurstSlc
        S1-A/B burst SLC object
    lut_par: dict
        Dictionary with LUT parameters
    dem_path: str
        File path to DEM
    tec_path: str
        File path to ionosphere TEC file
    scratch_path: str
        File path to scratch directory (default: None)
    weather_model_path: str
        File path to weather model file (default: None)

    Returns
    -------
    rg_lut: isce3.core.LUT2d
        Cumulative LUT in slant range direction (meters)
    az_lut: isce3.core.LUT2d
        Cumulative LUT in azimuth direction (seconds)
    '''
    # Dem info
    dem_raster = isce3.io.Raster(dem_path)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # Get LUT spacing
    rg_step = lut_par.range_spacing
    az_step = lut_par.azimuth_spacing

    # Create directory to temporary results
    output_path = f'{scratch_path}/corrections'
    os.makedirs(output_path, exist_ok=True)

    # If any of the following corrections is enabled
    # generate rdr2geo layers
    rdr2geo_enabled = lut_par.azimuth_fm_rate or \
                      lut_par.solid_earth_tides or \
                      lut_par.ionosphere_tec or \
                      lut_par.static_troposphere or \
                      lut_par.weather_model_troposphere
    if rdr2geo_enabled:
        # return contents: lon_path, lat_path, height_path, inc_path, head_path
        rdr2geo_raster_paths = compute_rdr2geo_rasters(burst, dem_raster,
                                                       output_path, rg_step,
                                                       az_step)
        # Open rdr2geo layers
        lon, lat, height, inc_angle, head_angle = \
            [open_raster(raster_path) for raster_path in rdr2geo_raster_paths]

    # Get the shape of the correction LUT and create empty numpy array
    lut = burst.bistatic_delay(range_step=rg_step,
                               az_step=az_step)
    lut_shape = lut.data.shape
    rg_data = np.zeros(lut_shape, dtype=np.float32)
    az_data = np.zeros(lut_shape, dtype=np.float32)

    # Initialize data list and lut_description to save corrections
    data_dict_key_dscrs = (
        ['doppler', 'Slant range geometry and steering doppler'],
        ['bistatic_delay', 'Bistatic delay'],
        ['azimuth_fm_rate', 'Azimuth FM rate mismatch'],
        ['rg_set', 'Slant range Solid Earth Tides'],
        ['az_set', 'Azimuth Solid Earth Tides'],
        ['static_tropo', 'Static troposphere delay'],
        ['tec_iono', 'Slant range ionosphere delay'],
        ['dry_tropo', 'Dry troposphere delay from weather model'],
        ['wet_tropo', 'Wet troposphere delay from weather model'])
    data_dict = {key: (np.zeros_like(rg_data), dscr)
                 for (key, dscr) in data_dict_key_dscrs}

    # Dict of meta correction items to be written to HDF5
    correction_lut_items = []

    # Common string to all lut_descriptions
    lut_desc = 'correction as a function of slant range and azimuth time'

    # Dict indicating if a correction item has been applied
    correction_application_items = []

    # Common string to all lut_descriptions
    corr_desc = 'correction has been applied'

    # Check which corrections are requested and accumulate corresponding data
    # Geometrical and steering doppler
    correction_application_items.append(
        Meta('geometry_steering_doppler', lut_par.geometry_steering_doppler,
             f'Boolean if geometry steering doppler {corr_desc}'))
    if lut_par.geometry_steering_doppler:
        doppler = burst.doppler_induced_range_shift(range_step=rg_step,
                                                    az_step=az_step)
        doppler_meter = doppler.data * isce3.core.speed_of_light * 0.5
        rg_data += doppler_meter
        correction_lut_items.append(
            Meta('geometry_steering_doppler', doppler_meter,
                 f'geometry steering doppler (range) {lut_desc}',
                 {'units': 'meters'}))

    # Bistatic delay
    correction_application_items.append(
        Meta('bistatic_delay', lut_par.bistatic_delay,
             f'Boolean if bistatic delay {corr_desc}'))
    if lut_par.bistatic_delay:
        bistatic_delay = burst.bistatic_delay(range_step=rg_step,
                                              az_step=az_step).data
        az_data -= bistatic_delay
        correction_lut_items.append(
            Meta('bistatic_delay', bistatic_delay,
                 f'bistatic delay (azimuth) {lut_desc}', {'units': 'seconds'}))

    # Azimuth FM-rate mismatch
    correction_application_items.append(
        Meta('azimuth_fm_rate_mismatch', lut_par.azimuth_fm_rate,
             f'Boolean if azimuth FM rate mismatch mitigation {corr_desc}'))
    if lut_par.azimuth_fm_rate:
        az_fm_rate = burst.az_fm_rate_mismatch_from_llh(lat, lon, height,
                                                        ellipsoid,
                                                        burst.as_isce3_radargrid(
                                                            az_step=az_step,
                                                            rg_step=rg_step)).data
        az_data -= az_fm_rate
        correction_lut_items.append(
            Meta('azimuth_fm_rate_mismatch', az_fm_rate,
                 f'azimuth FM rate mismatch mitigation (azimuth) {lut_desc}',
                 {'units': 'seconds'}))

    # Solid Earth tides
    correction_application_items.append(
        Meta('los_solid_earth_tides', lut_par.solid_earth_tides,
             f'Boolean if LOS solid Earth tides {corr_desc}'))
    correction_application_items.append(
        Meta('azimuth_solid_earth_tides', lut_par.solid_earth_tides,
             f'Boolean if azimuth solid Earth tides {corr_desc}'))
    if lut_par.solid_earth_tides:
        dec_factor = int(np.round(5000.0 / rg_step))
        dec_slice = np.s_[::dec_factor]
        rg_set_temp, az_set_temp = solid_earth_tides(burst, lat[dec_slice, dec_slice],
                                                     lon[dec_slice, dec_slice],
                                                     inc_angle[dec_slice, dec_slice],
                                                     head_angle[dec_slice, dec_slice])

        # Resize SET to the size of the correction grid
        kwargs = dict(order=1, mode='edge', anti_aliasing=True,
                      preserve_range=True)
        rg_set = resize(rg_set_temp, lut_shape, **kwargs)
        az_set = resize(az_set_temp, lut_shape, **kwargs)
        rg_data += rg_set
        az_data += az_set
        correction_lut_items.append(
            Meta('los_solid_earth_tides', rg_set,
                 f'Solid Earth tides (range) {lut_desc}', {'units': 'meters'}))
        correction_lut_items.append(
            Meta('azimuth_solid_earth_tides', az_set,
                 f'Solid Earth tides (azimuth) {lut_desc}', {'units': 'seconds'}))

    # Static troposphere
    correction_application_items.append(
        Meta('static_los_tropospheric_delay', lut_par.static_troposphere,
             f'Boolean if static tropospheric delay {corr_desc}'))
    if lut_par.static_troposphere:
        los_static_tropo = compute_static_troposphere_delay(inc_angle, height)
        rg_data += los_static_tropo
        correction_lut_items.append(
            Meta('static_los_tropospheric_delay', los_static_tropo,
                 f'Static tropospheric delay (range) {lut_desc}',
                 {'units': 'meters'}))

    # Ionosphere TEC correction
    correction_application_items.append(
        Meta('los_ionospheric_delay', lut_par.ionosphere_tec,
             f'Boolean if ionospheric delay {corr_desc}'))
    if lut_par.ionosphere_tec:
        los_iono = ionosphere_delay(burst.sensing_mid,
                                    burst.wavelength,
                                    tec_path, lon, lat, inc_angle)
        rg_data += los_iono
        correction_lut_items.append(
            Meta('los_ionospheric_delay', los_iono,
                 f'Ionospheric delay (range) {lut_desc}', {'units': 'meters'}))

    # Weather model troposphere correction
    tropo_enabled = lut_par.weather_model_troposphere.enabled
    delay_type = lut_par.weather_model_troposphere.delay_type
    correction_application_items.append(
        Meta('wet_los_troposphere_delay', tropo_enabled and 'wet' in delay_type,
             f'Boolean if wet LOS troposphere delay {corr_desc}'))
    correction_application_items.append(
        Meta('dry_los_troposphere_delay', tropo_enabled and 'dry' in delay_type,
             f'Boolean if dry LOS troposphere delay {corr_desc}'))
    if lut_par.weather_model_troposphere.enabled:
        from RAiDER.delay import tropo_delay
        from RAiDER.llreader import RasterRDR
        from RAiDER.losreader import Zenith

        # Instantiate an "aoi" object to read lat/lon/height files
        aoi = RasterRDR(rdr2geo_raster_paths[1], rdr2geo_raster_paths[0],
                        rdr2geo_raster_paths[2])

        # Instantiate the Zenith object. Note RAiDER LOS object requires
        # the orbit files.
        los = Zenith()

        # Compute the troposphere delay along the Zenith
        zen_wet, zen_dry = tropo_delay(burst.sensing_start,
                                       weather_model_path,
                                       aoi, los)

        # RaiDER delay is one-way only. Get the LOS delay my multiplying
        # by the incidence angle
        if 'wet' in delay_type:
            wet_los_tropo = 2.0 * zen_wet / np.cos(np.deg2rad(inc_angle))
            rg_data += wet_los_tropo
            correction_lut_items.append(
                Meta('wet_los_troposphere_delay', wet_los_tropo,
                     f'Wet LOS troposphere delay {lut_desc}',
                     {'units': 'meters'}))
        if 'dry' in delay_type:
            dry_los_tropo = 2.0 * zen_dry / np.cos(np.deg2rad(inc_angle))
            rg_data += dry_los_tropo
            correction_lut_items.append(
                Meta('dry_los_troposphere_delay', dry_los_tropo,
                     f'Dry LOS troposphere delay {lut_desc}',
                     {'units': 'meters'}))

    proc_nfo_group = \
             h5_file_obj.require_group('science/SENTINEL1/CSLC/metadata/processing_information/corrections')
    for meta_item in correction_application_items:
        add_dataset_and_attrs(proc_nfo_group, meta_item)

    correction_group = h5_file_obj.require_group('science/SENTINEL1/CSLC/corrections')
    for meta_item in correction_lut_items:
        add_dataset_and_attrs(correction_group, meta_item)

    # Create the range and azimuth LUT2d
    rg_lut = isce3.core.LUT2d(lut.x_start, lut.y_start,
                              lut.x_spacing, lut.y_spacing,
                              rg_data)
    az_lut = isce3.core.LUT2d(lut.x_start, lut.y_start,
                              lut.x_spacing, lut.y_spacing,
                              az_data)

    return rg_lut, az_lut


def solid_earth_tides(burst, lat_radar_grid, lon_radar_grid, inc_angle,
                      head_angle):
    '''
    Compute displacement due to Solid Earth Tides (SET)
    in slant range and azimuth directions

    Parameters
    ---------
    burst: Sentinel1Slc
        S1-A/B burst object
    lat_radar_grid: np.ndarray
        Latitude array on burst radargrid
    lon_radar_grid: np.ndarray
        Longitude array on burst radargrid
    inc_angle: np.ndarray
        Incident angle raster in unit of degrees
    head_angle: np.ndaaray
        Heading angle raster in unit of degrees

    Returns
    ------
    rg_set: np.ndarray
        2D array with SET displacement along LOS
    az_set: np.ndarray
        2D array with SET displacement along azimuth
    '''

    # Extract top-left coordinates from burst polygon
    lon_min, lat_min, _, _ = burst.border[0].bounds

    # Generate the atr object to run pySolid. We compute SET on a
    # 2.5 km x 2.5 km coarse grid
    margin = 0.1
    lat_start = lat_min - margin
    lon_start = lon_min - margin

    atr = {
        'LENGTH': 25,
        'WIDTH': 100,
        'X_FIRST': lon_start,
        'Y_FIRST': lat_start,
        'X_STEP': 0.023,
        'Y_STEP': 0.023
    }

    # Run pySolid and get SET in ENU coordinate system
    (set_e,
     set_n,
     set_u) = pysolid.calc_solid_earth_tides_grid(burst.sensing_start, atr,
                                                  display=False, verbose=True)

    # Resample SET from geographical grid to radar grid
    # Generate the lat/lon arrays for the SET geogrid
    lat_geo_array = np.linspace(atr['Y_FIRST'],
                                lat_start + atr['Y_STEP'] * atr['LENGTH'],
                                num=atr['LENGTH'])
    lon_geo_array = np.linspace(atr['X_FIRST'],
                                lon_start + atr['X_STEP'] * atr['WIDTH'],
                                num=atr['WIDTH'])

    # Use scipy RGI to resample SET from geocoded to radar coordinates
    pts_src = (np.flipud(lat_geo_array), lon_geo_array)
    pts_dst = (lat_radar_grid.flatten(), lon_radar_grid.flatten())

    rdr_set_e, rdr_set_n, rdr_set_u = \
        [resample_set(set_enu, pts_src, pts_dst).reshape(lat_radar_grid.shape)
         for set_enu in [set_e, set_n, set_u]]

    # Convert SET from ENU to range/azimuth coordinates
    # Note: rdr2geo heading angle is measured wrt to the East and it is positive
    # anti-clockwise. To convert ENU to LOS, we need the azimuth angle which is
    # measured from the north and positive anti-clockwise
    # azimuth_angle = heading + 90
    set_rg = enu2los(rdr_set_e, rdr_set_n, rdr_set_u, inc_angle,
                     az_angle=head_angle + 90.0)
    set_az = en2az(rdr_set_e, rdr_set_n, head_angle - 90.0)

    return set_rg, set_az


def compute_rdr2geo_rasters(burst, dem_raster, output_path,
                            rg_step, az_step):
    '''
    Get latitude, longitude, incidence and
    azimuth angle on multi-looked radar grid

    Parameters
    ----------
    burst: Sentinel1Slc
        S1-A/B burst object
    dem_raster: isce3.io.Raster
        ISCE3 object including DEM raster
    output_path: str
        Path where to save output rasters
    rg_step: float
        Spacing of radar grid along slant range
    az_step: float
        Spacing of the radar grid along azimuth

    Returns
    -------
    x_path: str
        Path to longitude raster
    y_path: str
        Path to latitude raster
    inc_path: str
        Path to incidence angle raster
    head_path: str
        Path to heading angle raster
    '''

    # Some ancillary inputs
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # Get radar grid for the correction grid
    rdr_grid = burst.as_isce3_radargrid(az_step=az_step,
                                        rg_step=rg_step)

    grid_doppler = isce3.core.LUT2d()

    # Initialize the rdr2geo object
    rdr2geo_obj = isce3.geometry.Rdr2Geo(rdr_grid, burst.orbit,
                                         ellipsoid, grid_doppler,
                                         threshold=1.0e-8)

    # Get the rdr2geo raster needed for SET computation
    topo_output = {f'{output_path}/x.rdr': gdal.GDT_Float64,
                   f'{output_path}/y.rdr': gdal.GDT_Float64,
                   f'{output_path}/height.rdr': gdal.GDT_Float64,
                   f'{output_path}/incidence_angle.rdr': gdal.GDT_Float32,
                   f'{output_path}/heading_angle.rdr': gdal.GDT_Float32}
    raster_list = [
        isce3.io.Raster(fname, rdr_grid.width,
                        rdr_grid.length, 1, dtype, 'ENVI')
        for fname, dtype in topo_output.items()]
    x_raster, y_raster, height_raster, incidence_raster, heading_raster = raster_list

    # Run rdr2geo on coarse radar grid
    rdr2geo_obj.topo(dem_raster, x_raster, y_raster,
                     height_raster=height_raster,
                     incidence_angle_raster=incidence_raster,
                     heading_angle_raster=heading_raster)

    # Return file path to rdr2geo layers
    paths = list(topo_output.keys())
    return paths[0], paths[1], paths[2], paths[3], paths[4]


def resample_set(geo_tide, pts_src, pts_dest):
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


def compute_static_troposphere_delay(incidence_angle_arr, hgt_arr):
    '''
    Compute troposphere delay using static model

    Parameters:
    -----------
    inc_path: str
        Path to incidence angle raster in radar grid in degrees
    hgt_path: str
        Path to surface heightraster in radar grid in meters

    Return:
    -------
    tropo: np.ndarray
        Troposphere delay in slant range
    '''
    ZPD = 2.3
    H = 6000.0

    tropo = ZPD / np.cos(np.deg2rad(incidence_angle_arr)) * np.exp(-1 * hgt_arr / H)

    return tropo

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
from compass.utils.iono import ionosphere_delay
from compass.utils.helpers import open_raster
from RAiDER.delay import tropo_delay
from RAiDER.llreader import RasterRDR
from RAiDER.losreader import Zenith


def correction_luts(burst, lut_par, dem_path, tec_path,
                    scratch_path=None,
                    weather_model_path=None):
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

    # Initialize data list and description to save corrections
    data_dict = {
        'doppler': [rg_data, 'Slant range geometry and steering doppler'],
        'bistatic_delay': [rg_data, 'Bistatic delay'],
        'azimuth_fm_rate': [rg_data, 'Azimuth FM rate mismatch'],
        'rg_set': [rg_data, 'Slant range Solid Earth Tides'],
        'az_set': [rg_data, 'Azimuth Solid Earth Tides'],
        'static_tropo': [rg_data, 'Static troposphere delay'],
        'tec_iono': [rg_data, 'Slant range ionosphere delay'],
        'dry_tropo': [rg_data, 'Dry troposphere delay from weather model'],
        'wet_tropo': [rg_data, 'Wet troposphere delay from weather model']}

    # Check which corrections are requested and accumulate corresponding data
    # Geometrical and steering doppler
    if lut_par.geometry_steering_doppler:
        doppler = burst.doppler_induced_range_shift(range_step=rg_step,
                                                    az_step=az_step)
        doppler_meter = doppler.data * isce3.core.speed_of_light * 0.5
        rg_data += doppler_meter
        data_dict['doppler'][0] = doppler_meter

    # Bistatic delay
    if lut_par.bistatic_delay:
        bistatic_delay = burst.bistatic_delay(range_step=rg_step,
                                              az_step=az_step).data
        az_data -= bistatic_delay
        data_dict['bistatic_delay'][0] = -bistatic_delay

    # Azimuth FM-rate mismatch
    if lut_par.azimuth_fm_rate:
        az_fm_rate = burst.az_fm_rate_mismatch_from_llh(lat, lon, height,
                                                        ellipsoid,
                                                        burst.as_isce3_radargrid(
                                                            az_step=az_step,
                                                            rg_step=rg_step)).data
        az_data -= az_fm_rate
        data_dict['azimuth_fm_rate'][0] = -az_fm_rate

    # Solid Earth tides
    if lut_par.solid_earth_tides:
        dec_factor = int(np.round(5000.0 / rg_step))
        dec_slice = np.s_[::dec_factor]
        rg_set_temp, az_set_temp = solid_earth_tides(burst, lat[dec_slice],
                                                     lon[dec_slice],
                                                     inc_angle[dec_slice],
                                                     head_angle[dec_slice])

        # Resize SET to the size of the correction grid
        kwargs = dict(order=1, mode='edge', anti_aliasing=True,
                      preserve_range=True)
        rg_set = resize(rg_set_temp, lut_shape, **kwargs)
        az_set = resize(az_set_temp, lut_shape, **kwargs)
        rg_data += rg_set
        az_data += az_set
        data_dict['rg_set'][0] = rg_set
        data_dict['az_set'][0] = az_set

    # Static troposphere
    if lut_par.static_troposphere:
        los_static_tropo = compute_static_troposphere_delay(inc_angle, height)
        rg_data += los_static_tropo
        data_dict['static_tropo'][0] = los_static_tropo

    # Ionosphere TEC correction
    if lut_par.ionosphere_tec:
        los_iono = ionosphere_delay(burst.sensing_mid,
                                    burst.wavelength,
                                    tec_path, lon, lat, inc_angle)
        rg_data += los_iono
        data_dict['tec_iono'][0] = los_iono

    # Weather model troposphere correction
    if lut_par.weather_model_troposphere.enabled:
        delay_type = lut_par.weather_model_troposphere.delay_type
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
        wet_los_tropo = 2.0 * zen_wet / np.cos(np.deg2rad(inc_angle))
        dry_los_tropo = 2.0 * zen_dry / np.cos(np.deg2rad(inc_angle))

        if 'wet' in delay_type:
            rg_data += wet_los_tropo
            data_dict['wet_tropo'][0] = wet_los_tropo
        if 'dry' in delay_type:
            rg_data += dry_los_tropo
            data_dict['dry_tropo'][0] = dry_los_tropo

    # Create the range and azimuth LUT2d
    rg_lut = isce3.core.LUT2d(lut.x_start, lut.y_start,
                              lut.x_spacing, lut.y_spacing,
                              rg_data)
    az_lut = isce3.core.LUT2d(lut.x_start, lut.y_start,
                              lut.x_spacing, lut.y_spacing,
                              az_data)

    # Save corrections
    driver = gdal.GetDriverByName('ENVI')
    out_ds = driver.Create(f'{output_path}/corrections',
                           lut_shape[1], lut_shape[0], len(data_dict),
                           gdal.GDT_Float32)
    band = 0
    for key in data_dict.keys():
        band += 1
        raster_band = out_ds.GetRasterBand(band)
        raster_band.SetDescription(data_dict[key][1])
        raster_band.WriteArray(data_dict[key][0])

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
                                         threshold=1.0e8)

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

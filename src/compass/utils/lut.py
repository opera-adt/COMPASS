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

from compass.utils.geometry_utils import enu2rgaz
from compass.utils.iono import ionosphere_delay
from compass.utils.helpers import open_raster
from compass.utils.helpers import write_raster


def cumulative_correction_luts(burst, dem_path, tec_path,
                               scratch_path=None,
                               weather_model_path=None,
                               rg_step=200, az_step=0.25,
                               delay_type='dry',
                               geo2rdr_params=None):
    '''
    Sum correction LUTs and returns cumulative correction LUT in slant range
    and azimuth directions

    Parameters
    ----------
    burst: Sentinel1BurstSlc
        Sentinel-1 A/B burst SLC object
    dem_path: str
        Path to the DEM file
    tec_path: str
        Path to the TEC file in IONEX format
    scratch_path: str
        Path to the scratch directory
    weather_model_path: str
        Path to the weather model file in NetCDF4 format.
        This file has been preprocessed by RAiDER and it is
        the only file format supported by the package. If None,
        no troposphere correction is performed.
    rg_step: float
        LUT spacing along slant range direction
    az_step: float
        LUT spacing along azimuth direction
    delay_type: str
        Type of troposphere delay. Any between 'dry', or 'wet', or
        'wet_dry' for the sum of wet and dry troposphere delays.

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
    geometrical_steer_doppler, bistatic_delay, az_fm_mismatch, [tide_rg, tide_az], \
        los_ionosphere, [wet_los_tropo, dry_los_tropo], los_static_tropo = \
        compute_geocoding_correction_luts(burst,
                                          dem_path=dem_path,
                                          tec_path=tec_path,
                                          scratch_path=scratch_path,
                                          weather_model_path=weather_model_path,
                                          rg_step=rg_step,
                                          az_step=az_step,
                                          geo2rdr_params=geo2rdr_params)

    # Convert to geometrical doppler from range time (seconds) to range (m)
    geometry_doppler = geometrical_steer_doppler.data * isce3.core.speed_of_light * 0.5
    rg_lut_data = geometry_doppler + tide_rg + los_ionosphere + los_static_tropo

    # Add troposphere delay to range LUT
    if 'wet' in delay_type:
        rg_lut_data += wet_los_tropo
    if 'dry' in delay_type:
        rg_lut_data += dry_los_tropo

    # Invert signs to correct for convention
    #az_lut_data = -(bistatic_delay.data + az_fm_mismatch.data)
    az_lut_data = -bistatic_delay.data
    # NOTE: Azimuth FM rate was turned off for OPERA production

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
    data_list = [geometry_doppler, bistatic_delay.data, az_fm_mismatch.data,
                 tide_rg, tide_az, los_ionosphere]
    descr = ['slant range geometrical doppler', 'azimuth bistatic delay',
             'azimuth FM rate mismatch', 'slant range Solid Earth tides',
             'azimuth time Solid Earth tides', 'line-of-sight ionospheric delay']

    if weather_model_path is not None:
        if 'wet' in delay_type:
            data_list.append(wet_los_tropo)
            descr.append('wet LOS troposphere')
        if 'dry' in delay_type:
            data_list.append(dry_los_tropo)
            descr.append('dry LOS troposphere')

    write_raster(f'{output_path}/corrections', data_list, descr)

    return rg_lut, az_lut


def compute_geocoding_correction_luts(burst, dem_path, tec_path,
                                      scratch_path=None,
                                      weather_model_path=None,
                                      rg_step=200, az_step=0.25,
                                      geo2rdr_params=None):
    '''
    Compute slant range and azimuth LUTs corrections
    to be applied during burst geocoding

    Parameters
    ----------
    burst: Sentinel1BurstSlc
        S1-A/B burst object
    dem_path: str
        Path to the DEM required for azimuth FM rate mismatch.
    tec_path: str
        Path to the TEC file for ionosphere correction
    scratch_path: str
        Path to the scratch directory.
        If `None`, `burst.az_fm_rate_mismatch_mitigation()` will
        create temporary directory internally.
    weather_model_path: str
        Path to troposphere weather model in NetCDF4 format.
        This is the only format supported by RAiDER. If None,
        no weather model-based troposphere correction is applied
        (default: None).
    rg_step: int
        LUT spacing along slant range in meters
    az_step: int
        LUT spacing along azimuth in seconds

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

    [rg_set, az_set]: list[np.ndarray]
        List of numpy.ndarray containing SET in slant range and azimuth directions
        in meters. These corrections need to be added to the slC tagged azimuth
        and slant range times.

    ionosphere: np.ndarray
        numpy.ndarray for ionosphere delay in line-of-sight direction in meters.
        This correction needs to be added to the SLC tagged range time to
        get the corrected range times.
    [wet_los_tropo, dry_los_tropo]: list[np.ndarray]
        List of numpy.ndarray containing the LOS wet and dry troposphere delays
        computed from the file specified under 'weather_model_path'. These delays
        need to be added to the slant range correction LUT2D.
    '''

    # Get DEM raster
    dem_raster = isce3.io.Raster(dem_path)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # Create directory to store SET temp results
    output_path = f'{scratch_path}/corrections'
    os.makedirs(output_path, exist_ok=True)

    # Compute Geometrical Steering Doppler
    geometrical_steering_doppler = \
        burst.doppler_induced_range_shift(range_step=rg_step, az_step=az_step)

    # Compute bistatic delay
    bistatic_delay = burst.bistatic_delay(range_step=rg_step, az_step=az_step)

    # Run rdr2geo to obtain the required layers
    # return contents: lon_path, lat_path, height_path, inc_path, head_path
    rdr2geo_raster_paths = compute_rdr2geo_rasters(burst, dem_raster,
                                                   output_path, rg_step,
                                                   az_step)

    # Open rdr2geo layers
    lon, lat, height, inc_angle, head_angle = \
        [open_raster(raster_path) for raster_path in rdr2geo_raster_paths]

    # Compute azimuth FM-rate mismatch
    az_fm_mismatch = burst.az_fm_rate_mismatch_from_llh(lat, lon, height,
                                                        ellipsoid,
                                                        burst.as_isce3_radargrid(
                                                            az_step=az_step,
                                                            rg_step=rg_step)
                                                        )

    # compute Solid Earth Tides using pySolid. Decimate the rdr2geo layers.
    # compute decimation factor assuming a 5 km spacing along slant range
    dec_factor = int(np.round(5000.0 / rg_step))
    dec_slice = np.s_[::dec_factor, ::dec_factor]
    rg_set_temp, az_set_temp = solid_earth_tides(burst,
                                                 lat[dec_slice],
                                                 lon[dec_slice],
                                                 height[dec_slice],
                                                 ellipsoid,
                                                 geo2rdr_params)
    out_shape = bistatic_delay.data.shape
    kwargs = dict(order=1, mode='edge', anti_aliasing=True,
                    preserve_range=True)
    rg_set = resize(rg_set_temp, out_shape, **kwargs)
    az_set = resize(az_set_temp, out_shape, **kwargs)

    # Compute ionosphere delay
    los_ionosphere = ionosphere_delay(burst.sensing_mid,
                                      burst.wavelength,
                                      tec_path, lon, lat, inc_angle)

    # Compute wet and dry troposphere delays using RAiDER
    wet_los_tropo, dry_los_tropo, los_static_tropo =\
        [np.zeros(out_shape) for _ in range(3)]

    if weather_model_path is None:
        # Compute static troposphere correction
        los_static_tropo = compute_static_troposphere_delay(inc_angle, height)

    else:
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
        wet_los_tropo = 2.0 * zen_wet / np.cos(np.deg2rad(inc_angle))
        dry_los_tropo = 2.0 * zen_dry / np.cos(np.deg2rad(inc_angle))

    return (
        geometrical_steering_doppler, bistatic_delay, az_fm_mismatch,
        [rg_set, az_set], los_ionosphere,
        [wet_los_tropo, dry_los_tropo], los_static_tropo
    )


def solid_earth_tides(burst, lat_radar_grid, lon_radar_grid, hgt_radar_grid,
                      ellipsoid, geo2rdr_params=None):
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
    set_rg, set_az = enu2rgaz(burst.as_isce3_radargrid(), burst.orbit, ellipsoid,
             lon_radar_grid, lat_radar_grid, hgt_radar_grid,
             rdr_set_e, rdr_set_n, rdr_set_u, geo2rdr_params)

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

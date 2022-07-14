import os
import time

import isce3
import journal
import numpy as np
from osgeo import gdal

from compass.utils.rtc_metadata import RtcMetadata
from compass.utils.geo_runconfig import GeoRunConfig
from compass.utils.range_split_spectrum import range_split_spectrum
from compass.utils.yaml_argparse import YamlArgparse

from osgeo import osr,gdal
from s1reader.s1_burst_slc import Sentinel1BurstSlc

# TODO: remove PLAnT mosaicking and bands merging
import plant
def _mosaic(input_files, output_file):
    plant.mosaic(input_files, output_file = output_file,
                 no_average=True, force=True, interp='average')
def _merge_bands(input_files, output_file):
    plant.util(input_files, output_file = output_file, force=True)

def snap_coord(val, snap, round_func):
    snapped_value = round_func(float(val) / snap) * snap
    return snapped_value

def _update_mosaic_boundaries(mosaic_geogrid_dict, geogrid):
    xf = geogrid.start_x + geogrid.spacing_x * geogrid.width
    yf = geogrid.start_y + geogrid.spacing_y * geogrid.length
    if ('x0' not in mosaic_geogrid_dict.keys() or
            geogrid.start_x < mosaic_geogrid_dict['x0']):
        mosaic_geogrid_dict['x0'] = geogrid.start_x
    if ('xf' not in mosaic_geogrid_dict.keys() or
            xf > mosaic_geogrid_dict['xf']):
        mosaic_geogrid_dict['xf'] = xf
    if ('y0' not in mosaic_geogrid_dict.keys() or
            geogrid.start_y > mosaic_geogrid_dict['y0']):
        mosaic_geogrid_dict['y0'] = geogrid.start_y
    if ('yf' not in mosaic_geogrid_dict.keys() or
            yf < mosaic_geogrid_dict['yf']):
        mosaic_geogrid_dict['yf'] = yf
    if 'dx' not in mosaic_geogrid_dict.keys():
        mosaic_geogrid_dict['dx'] = geogrid.spacing_x
    else:
        assert(mosaic_geogrid_dict['dx'] == geogrid.spacing_x)
    if 'dy' not in mosaic_geogrid_dict.keys():
        mosaic_geogrid_dict['dy'] = geogrid.spacing_y
    else:
        assert(mosaic_geogrid_dict['dy'] == geogrid.spacing_y)
    if 'epsg' not in mosaic_geogrid_dict.keys():
        mosaic_geogrid_dict['epsg'] = geogrid.epsg
    else:
        assert(mosaic_geogrid_dict['epsg'] == geogrid.epsg)


def _get_raster(output_dir, ds_name, dtype, shape,
                output_file_list, output_obj_list, 
                flag_save_vector_1):
    if flag_save_vector_1 is not True:
        return None

    output_file = os.path.join(output_dir, ds_name)+'.tif'
    raster_obj = isce3.io.Raster(
        output_file,
        shape[2],
        shape[1],
        shape[0],
        dtype,
        "GTiff")
    output_file_list.append(output_file)
    output_obj_list.append(raster_obj)
    return raster_obj


def _add_output_to_output_metadata_dict(flag, key, output_dir,
        output_metadata_dict):
    if not flag:
        return
    output_image_list = []
    output_metadata_dict[key] = \
        [os.path.join(output_dir, f'rtc_product_{key}.tif'), output_image_list]



def thermal_correction(burst_in: Sentinel1BurstSlc,
                       path_slc_vrt: str,
                       path_slc_out: str,
                       flag_output_complex: bool = False):
    '''Apply thermal correction stored in burst_in. Save the corrected signal back to ENVI format. Preserves the phase.'''

    # Load the SLC of the burst
    burst_in.slc_to_vrt_file(path_slc_vrt)
    raster_slc_from = gdal.Open(path_slc_vrt)
    arr_slc_from = raster_slc_from.ReadAsArray()

    # Generate the correction layer
    arr_noise = burst_in.burst_noise.export_lut()

    # Apply the correction
    corrected_image = np.abs(arr_slc_from) ** 2 - arr_noise
    min_backscatter = 0
    max_backscatter = None
    corrected_image = np.clip(corrected_image, min_backscatter,
                              max_backscatter)
    if flag_output_complex:
        factor_mag = np.sqrt(corrected_image) / np.abs(arr_slc_from)
        factor_mag[np.isnan(factor_mag)] = 0.0
        corrected_image = arr_slc_from * factor_mag
        dtype = gdal.GDT_CFloat32
    else:
        dtype = gdal.GDT_Float32

    # Save the corrected image
    drvout = gdal.GetDriverByName('ENVI')
    raster_out = drvout.Create(path_slc_out, burst_in.shape[1],
                               burst_in.shape[0], 1, dtype)
    band_out = raster_out.GetRasterBand(1)
    band_out.WriteArray(corrected_image)
    band_out.FlushCache()
    del band_out


def run(cfg):
    '''
    Run geocode burst workflow with user-defined
    args stored in dictionary runconfig *cfg*

    Parameters
    ---------
    cfg: dict
        Dictionary with user runconfig options
    '''
    info_channel = journal.info("rtc.run")

    # Start tracking processing time
    t_start = time.time()
    info_channel.log("Starting geocode burst")


    dem_interp_method_enum = cfg.groups.processing.dem_interpolation_method_enum



    product_path = cfg.groups.product_path_group.product_path
    scratch_path = cfg.groups.product_path_group.scratch_path
    output_dir = cfg.groups.product_path_group.sas_output_dir


    # unpack geocode run parameters
    geocode_namespace = cfg.groups.processing.geocoding
    geocode_algorithm = geocode_namespace.algorithm_type
    output_mode = geocode_namespace.output_mode
    flag_apply_rtc = geocode_namespace.apply_rtc
    flag_apply_thermal_noise_correction = \
        geocode_namespace.apply_thermal_noise_correction
    memory_mode = geocode_namespace.memory_mode
    geogrid_upsampling = geocode_namespace.geogrid_upsampling
    abs_cal_factor = geocode_namespace.abs_rad_cal
    clip_max = geocode_namespace.clip_max
    clip_min = geocode_namespace.clip_min
    # geogrids = geocode_namespace.geogrids
    flag_upsample_radar_grid = geocode_namespace.upsample_radargrid
    flag_save_incidence_angle = geocode_namespace.save_incidence_angle
    flag_save_local_inc_angle = geocode_namespace.save_local_inc_angle
    flag_save_projection_angle = geocode_namespace.save_projection_angle
    flag_save_simulated_radar_brightness = \
        geocode_namespace.save_simulated_radar_brightness
    flag_save_directional_slope_angle = \
        geocode_namespace.save_directional_slope_angle
    flag_save_nlooks = geocode_namespace.save_nlooks
    flag_save_rtc = geocode_namespace.save_rtc
    flag_save_dem = geocode_namespace.save_dem

    # unpack RTC run parameters
    rtc_namespace = cfg.groups.processing.rtc
    output_terrain_radiometry = rtc_namespace.output_type
    rtc_algorithm = rtc_namespace.algorithm_type
    input_terrain_radiometry = rtc_namespace.input_terrain_radiometry
    rtc_min_value_db = rtc_namespace.rtc_min_value_db
    rtc_upsampling = rtc_namespace.dem_upsampling


    # Common initializations
    dem_raster = isce3.io.Raster(cfg.dem)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid
    zero_doppler = isce3.core.LUT2d()
    threshold = cfg.geo2rdr_params.threshold
    maxiter = cfg.geo2rdr_params.numiter
    exponent = 1 if flag_apply_thermal_noise_correction else 2

    # output mosaics
    geo_filename = f'{output_dir}/'f'rtc_product.tif'
    output_imagery_dict = {}
    output_metadata_dict = {}

    _add_output_to_output_metadata_dict(flag_save_nlooks, 'nlooks', output_dir,
                                       output_metadata_dict)
    _add_output_to_output_metadata_dict(flag_save_rtc, 'rtc', output_dir,
                                       output_metadata_dict)
    '''
    _add_output_to_output_metadata_dict(flag_save_dem, 'interpolated_dem',
                                        output_dir, output_metadata_dict)
    '''

    mosaic_geogrid_dict = {}
    temp_files_list = []

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(scratch_path, exist_ok=True)

    # iterate over sub-burts
    for burst in cfg.bursts:

        t_burst_start = time.time()

        date_str = burst.sensing_start.strftime("%Y%m%d")
        burst_id = burst.burst_id
        info_channel.log(f'processing burst: {burst_id}')
        pol = burst.polarization
        geogrid = cfg.geogrids[burst_id]

        # snap coordinates
        x_snap = geogrid.spacing_x
        y_snap = geogrid.spacing_y
        geogrid.start_x = snap_coord(geogrid.start_x, x_snap, np.floor)
        geogrid.start_y = snap_coord(geogrid.start_y, y_snap, np.ceil)

        # update mosaic boundaries
        _update_mosaic_boundaries(mosaic_geogrid_dict, geogrid)

        burst_scratch_path = f'{scratch_path}/{burst_id}/{date_str}'
        os.makedirs(burst_scratch_path, exist_ok=True)

        radar_grid = burst.as_isce3_radargrid()
        # native_doppler = burst.doppler.lut2d
        orbit = burst.orbit
        if 'orbit' not in mosaic_geogrid_dict.keys():
            mosaic_geogrid_dict['orbit'] = orbit
        if 'wavelength' not in mosaic_geogrid_dict.keys():
            mosaic_geogrid_dict['wavelength'] = burst.wavelength
        if 'lookside' not in mosaic_geogrid_dict.keys():
            mosaic_geogrid_dict['lookside'] = radar_grid.lookside

        temp_slc_path = f'{scratch_path}/{burst_id}_{pol}_temp.vrt'
        temp_slc_corrected_path=f'{scratch_path}/{burst_id}_{pol}_corrected_temp'
        burst.slc_to_vrt_file(temp_slc_path)
        if not flag_apply_thermal_noise_correction:
            rdr_burst_raster = isce3.io.Raster(temp_slc_path)
            temp_files_list.append(temp_slc_path)
        else:
            thermal_correction(burst,temp_slc_path,temp_slc_corrected_path)
            rdr_burst_raster = isce3.io.Raster(temp_slc_corrected_path)

        # Generate output geocoded burst raster
        geo_burst_filename = (f'{scratch_path}/'
                              f'{burst_id}_{date_str}_{pol}.tif')

        geo_burst_raster = isce3.io.Raster(
            geo_burst_filename,
            geogrid.width, geogrid.length,
            rdr_burst_raster.num_bands, gdal.GDT_Float32,
            cfg.geocoding_params.output_format)
        temp_files_list.append(geo_burst_filename)

        # init Geocode object depending on raster type
        if rdr_burst_raster.datatype() == gdal.GDT_Float32:
            geo_obj = isce3.geocode.GeocodeFloat32()
        elif rdr_burst_raster.datatype() == gdal.GDT_Float64:
            geo_obj = isce3.geocode.GeocodeFloat64()
        elif rdr_burst_raster.datatype() == gdal.GDT_CFloat32:
            geo_obj = isce3.geocode.GeocodeCFloat32()
        elif rdr_burst_raster.datatype() == gdal.GDT_CFloat64:
            geo_obj = isce3.geocode.GeocodeCFloat64()
        else:
            err_str = 'Unsupported raster type for geocoding'
            raise NotImplementedError(err_str)

        # init geocode members
        geo_obj.orbit = orbit
        geo_obj.ellipsoid = ellipsoid
        geo_obj.doppler = zero_doppler
        geo_obj.threshold_geo2rdr = threshold
        geo_obj.numiter_geo2rdr = maxiter

        # set data interpolator based on the geocode algorithm
        if output_mode == isce3.geocode.GeocodeOutputMode.INTERP:
            geo_obj.data_interpolator = geocode_algorithm

        geo_obj.geogrid(geogrid.start_x, geogrid.start_y,
                        geogrid.spacing_x, geogrid.spacing_y,
                        geogrid.width, geogrid.length, geogrid.epsg)

        if flag_save_nlooks:
            temp_nlooks = (f'{scratch_path}/'
                           f'{burst_id}_{date_str}_{pol}_nlooks.tif')
            out_geo_nlooks_obj = isce3.io.Raster(
                temp_nlooks,
                geogrid.width, geogrid.length, 1,
                gdal.GDT_Float32, cfg.geocoding_params.output_format)
            temp_files_list.append(temp_nlooks)
        else:
            temp_nlooks = None
            out_geo_nlooks_obj = None

        if flag_save_rtc:
            temp_rtc = (f'{scratch_path}/'
                           f'{burst_id}_{date_str}_{pol}_rtc_anf.tif')
            out_geo_rtc_obj = isce3.io.Raster(
                temp_rtc,
                geogrid.width, geogrid.length, 1,
                gdal.GDT_Float32, cfg.geocoding_params.output_format)
            temp_files_list.append(temp_rtc)
        else:
            temp_rtc = None
            out_geo_rtc_obj = None

        '''
        if flag_save_dem:
            temp_interpolated_dem = (f'{scratch_path}/'
                           f'{burst_id}_{date_str}_{pol}_interpolated_dem.tif')
            if (output_mode == 
                    isce3.geocode.GeocodeOutputMode.AREA_PROJECTION):
                interpolated_dem_width = geogrid.width + 1
                interpolated_dem_length = geogrid.length + 1
            else:
                interpolated_dem_width = geogrid.width
                interpolated_dem_length = geogrid.length
            out_geo_dem_obj = isce3.io.Raster(
                temp_interpolated_dem, 
                interpolated_dem_width, 
                interpolated_dem_length, 1,
                gdal.GDT_Float32, cfg.geocoding_params.output_format)
        else:
            temp_interpolated_dem = None
            out_geo_dem_obj = None
        '''

        # Extract burst boundaries and create sub_swaths object to mask
        # invalid radar samples
        sub_swaths = isce3.product.SubSwaths(1)
        valid_samples_sub_swath = np.repeat(
            [[burst.first_valid_sample, burst.last_valid_sample]],
            burst.last_valid_line - burst.first_valid_line, axis=0)
        sub_swaths.valid_samples_sub_swath(1, valid_samples_sub_swath)

        # geocode
        geo_obj.geocode(radar_grid=radar_grid,
                        input_raster=rdr_burst_raster,
                        output_raster=geo_burst_raster,
                        dem_raster=dem_raster,
                        output_mode=output_mode,
                        geogrid_upsampling=geogrid_upsampling,
                        flag_apply_rtc=flag_apply_rtc,
                        input_terrain_radiometry=input_terrain_radiometry,
                        output_terrain_radiometry=output_terrain_radiometry,
                        exponent=exponent,
                        rtc_min_value_db=rtc_min_value_db,
                        rtc_upsampling=rtc_upsampling,
                        rtc_algorithm=rtc_algorithm,
                        abs_cal_factor=abs_cal_factor,
                        flag_upsample_radar_grid=flag_upsample_radar_grid,
                        clip_min = clip_min,
                        clip_max = clip_max,
                        # radargrid_nlooks=radar_grid_nlooks,
                        # out_off_diag_terms=out_off_diag_terms_obj,
                        out_geo_nlooks=out_geo_nlooks_obj,
                        out_geo_rtc=out_geo_rtc_obj,
                        # out_geo_dem=out_geo_dem_obj,
                        input_rtc=None,
                        output_rtc=None,
                        dem_interp_method=dem_interp_method_enum,
                        memory_mode=memory_mode,
                        sub_swaths=sub_swaths)

        del geo_burst_raster
        info_channel.log(f'file saved: {geo_burst_filename}')
        if pol not in output_imagery_dict:
            output_imagery_dict[pol] = []
        output_imagery_dict[pol].append(geo_burst_filename)

        if flag_save_nlooks:
            del out_geo_nlooks_obj
            info_channel.log(f'file saved: {temp_nlooks}')
            output_metadata_dict['nlooks'][1].append(temp_nlooks)
    
        if flag_save_rtc:
            del out_geo_rtc_obj
            info_channel.log(f'file saved: {temp_rtc}')
            output_metadata_dict['rtc'][1].append(temp_rtc)

        '''
        if flag_save_dem:
            del out_geo_dem_obj
            info_channel.log(f'file saved: {temp_interpolated_dem}')
            output_metadata_dict['interpolated_dem'][1].append(
                temp_interpolated_dem)
        '''

        t_burst_end = time.time()
        info_channel.log(f'elapsed time (burst): {t_burst_end - t_burst_start}')

    # create mosaic geogrid
    if (flag_save_incidence_angle or flag_save_local_inc_angle or
            flag_save_projection_angle or
            flag_save_simulated_radar_brightness or flag_save_dem or
            flag_save_directional_slope_angle):
        mosaic_width = int(np.round((mosaic_geogrid_dict['xf'] -
                                     mosaic_geogrid_dict['x0']) /
                           mosaic_geogrid_dict['dx']))
        mosaic_length = int(np.round((mosaic_geogrid_dict['yf'] -
                                      mosaic_geogrid_dict['y0']) /
                            mosaic_geogrid_dict['dy']))
        mosaic_geogrid = isce3.product.GeoGridParameters(
            mosaic_geogrid_dict['x0'], mosaic_geogrid_dict['y0'],
            mosaic_geogrid_dict['dx'], mosaic_geogrid_dict['dy'],
            mosaic_width, mosaic_length,
            mosaic_geogrid_dict['epsg'])

        output_file_list = []
        output_obj_list = []
        layers_nbands = 1
        shape = [layers_nbands, mosaic_length, mosaic_width]

        incidence_angle_raster = _get_raster(
            output_dir, 'incidence_angle', gdal.GDT_Float32, shape, 
            output_file_list, output_obj_list, flag_save_incidence_angle)
        local_incidence_angle_raster = _get_raster(
            output_dir, 'local_incidence_angle', gdal.GDT_Float32, shape, 
            output_file_list, output_obj_list, flag_save_local_inc_angle)
        projection_angle_raster = _get_raster(
            output_dir, 'projection_angle', gdal.GDT_Float32, shape, 
            output_file_list, output_obj_list, flag_save_projection_angle)
        simulated_radar_brightness_raster = _get_raster(
            output_dir, 'simulated_radar_brightness', gdal.GDT_Float32, shape,
            output_file_list, output_obj_list,
            flag_save_simulated_radar_brightness)
        directional_slope_angle_raster = _get_raster(
            output_dir, 'directional_slope_angle', gdal.GDT_Float32, shape,
            output_file_list, output_obj_list,
            flag_save_directional_slope_angle)
        interpolated_dem_raster = _get_raster(
            output_dir, 'interpolated_dem', gdal.GDT_Float32, shape,
            output_file_list, output_obj_list, flag_save_dem)

        # TODO review this (Doppler)!!!
        # native_doppler = burst.doppler.lut2d
        native_doppler = isce3.core.LUT2d()
        native_doppler.bounds_error = False
        grid_doppler = isce3.core.LUT2d()
        grid_doppler.bounds_error = False

        # call get_radar_grid()
        isce3.geogrid.get_radar_grid(mosaic_geogrid_dict['lookside'],
                                     mosaic_geogrid_dict['wavelength'],
                                     dem_raster,
                                     mosaic_geogrid,
                                     orbit,
                                     native_doppler,
                                     grid_doppler,
                                     incidence_angle_raster =
                                        incidence_angle_raster,
                                     local_incidence_angle_raster =
                                        local_incidence_angle_raster,
                                    projection_angle_raster =
                                        projection_angle_raster,
                                    simulated_radar_brightness_raster =
                                        simulated_radar_brightness_raster,
                                    directional_slope_angle_raster =
                                        directional_slope_angle_raster,
                                    interpolated_dem_raster =
                                        interpolated_dem_raster,
                                    dem_interp_method=dem_interp_method_enum)
        '''
                                     # epsg_los_and_along_track_vectors,
                                     # interpolated_dem_raster,
                                     # slant_range_raster,
                                     # azimuth_time_raster,
                                     # incidence_angle_raster,
                                     # los_unit_vector_x_raster,
                                     # los_unit_vector_y_raster,
                                     # along_track_unit_vector_x_raster,
                                     # along_track_unit_vector_y_raster,
                                     # elevation_angle_raster,
                                     # ground_track_velocity_raster,
        '''
        '''
                                     # projection_angle_raster,
                                     # simulated_radar_brightness_raster,
                                     # dem_interp_method,
                                     # args.threshold_geo2rdr,
                                     # args.num_iter_geo2rdr,
                                     # args.delta_range_geo2rdr)
        '''
        # Flush data
        for obj in output_obj_list:
            del obj
        for filename in output_file_list:
            info_channel.log(f'file saved: {filename}')

    # mosaic sub-bursts
    mosaic_pol_list = []
    for pol in output_imagery_dict.keys():
        geo_pol_filename = (f'{output_dir}/'
                            f'rtc_product_{pol}.tif')
        imagery_list = output_imagery_dict[pol]
        info_channel.log(f'mosaicking file: {geo_pol_filename}')
        _mosaic(imagery_list, geo_pol_filename)
        mosaic_pol_list.append(geo_pol_filename)
    _merge_bands(mosaic_pol_list, geo_filename)
    output_file_list.append(geo_filename)

    # mosaic other bands
    for key in output_metadata_dict.keys():
        output_file, input_files = output_metadata_dict[key]
        info_channel.log(f'mosaicking file: {output_file}')
        _mosaic(input_files, output_file)
        output_file_list.append(output_file)

    info_channel.log('removing temporary files:')
    for filename in temp_files_list:
        if not os.path.isfile(filename):
            continue
        os.remove(filename)
        info_channel.log(f'    {filename}')

    info_channel.log('output files:')
    for filename in output_file_list:
        info_channel.log(f'    {filename}')

    t_end = time.time()
    info_channel.log(f'elapsed time: {t_end - t_start}')


def _load_parameters(cfg):
    '''
    Load GCOV specific parameters.
    '''
    error_channel = journal.error('gcov_runconfig.load')

    geocode_namespace = cfg.groups.processing.geocoding
    rtc_namespace = cfg.groups.processing.rtc

    if geocode_namespace.abs_rad_cal is None:
        geocode_namespace.abs_rad_cal = 1.0

    if geocode_namespace.clip_max is None:
        geocode_namespace.clip_max = np.nan

    if geocode_namespace.clip_min is None:
        geocode_namespace.clip_min = np.nan

    if geocode_namespace.geogrid_upsampling is None:
        geocode_namespace.geogrid_upsampling = 1.0

    if geocode_namespace.memory_mode == 'single_block':
        geocode_namespace.memory_mode = isce3.core.GeocodeMemoryMode.SingleBlock
    elif geocode_namespace.memory_mode == 'geogrid':
        geocode_namespace.memory_mode = isce3.core.GeocodeMemoryMode.BlocksGeogrid
    elif geocode_namespace.memory_mode == 'geogrid_and_radargrid':
        geocode_namespace.memory_mode = isce3.core.GeocodeMemoryMode.BlocksGeogridAndRadarGrid
    elif geocode_namespace.memory_mode == 'auto' or (geocode_namespace.memory_mode is None):
        geocode_namespace.memory_mode = isce3.core.GeocodeMemoryMode.Auto
    else:
        err_msg = f"ERROR memory_mode: {geocode_namespace.memory_mode}"
        raise ValueError(err_msg)

    rtc_output_type = rtc_namespace.output_type
    if rtc_output_type == 'sigma0':
        rtc_namespace.output_type = isce3.geometry.RtcOutputTerrainRadiometry.SIGMA_NAUGHT
    else:
        rtc_namespace.output_type = isce3.geometry.RtcOutputTerrainRadiometry.GAMMA_NAUGHT


    geocode_algorithm = cfg.groups.processing.geocoding.algorithm_type
    if geocode_algorithm == "area_projection":
        output_mode = isce3.geocode.GeocodeOutputMode.AREA_PROJECTION
    else:
        output_mode = isce3.geocode.GeocodeOutputMode.INTERP
    geocode_namespace.output_mode = output_mode

    # only 2 RTC algorithms supported: area_projection (default) & bilinear_distribution
    if rtc_namespace.algorithm_type == "bilinear_distribution":
        rtc_namespace.algorithm_type = isce3.geometry.RtcAlgorithm.RTC_BILINEAR_DISTRIBUTION
    else:
        rtc_namespace.algorithm_type = isce3.geometry.RtcAlgorithm.RTC_AREA_PROJECTION

    if rtc_namespace.input_terrain_radiometry == "sigma0":
        rtc_namespace.input_terrain_radiometry = isce3.geometry.RtcInputTerrainRadiometry.SIGMA_NAUGHT_ELLIPSOID
    else:
        rtc_namespace.input_terrain_radiometry = isce3.geometry.RtcInputTerrainRadiometry.BETA_NAUGHT

    if rtc_namespace.rtc_min_value_db is None:
        rtc_namespace.rtc_min_value_db = np.nan

    # Update the DEM interpolation method
    dem_interp_method = \
        cfg.groups.processing.dem_interpolation_method

    if dem_interp_method == 'biquintic':
        dem_interp_method_enum = isce3.core.DataInterpMethod.BIQUINTIC
    elif (dem_interp_method == 'sinc'):
        dem_interp_method_enum = isce3.core.DataInterpMethod.SINC
    elif (dem_interp_method == 'bilinear'):
        dem_interp_method_enum = isce3.core.DataInterpMethod.BILINEAR
    elif (dem_interp_method == 'bicubic'):
        dem_interp_method_enum = isce3.core.DataInterpMethod.BICUBIC
    elif (dem_interp_method == 'nearest'):
        dem_interp_method_enum = isce3.core.DataInterpMethod.NEAREST
    else:
        err_msg = ('ERROR invalid DEM interpolation method:'
                   f' {dem_interp_method}')
        raise ValueError(err_msg)

    cfg.groups.processing.dem_interpolation_method_enum = \
        dem_interp_method_enum

if __name__ == "__main__":
    '''Run geocode rtc workflow from command line'''
    # load arguments from command line
    geo_parser = YamlArgparse()

    # Get a runconfig dict from command line argumens
    cfg = GeoRunConfig.load_from_yaml(geo_parser.run_config_path,
                                      'rtc_s1')

    _load_parameters(cfg)

    # Run geocode burst workflow
    run(cfg)

    # Save burst metadata
    '''
    metadata = RtcMetadata.from_georunconfig(cfg)
    for burst in cfg.bursts:
        burst_id = burst.burst_id
        date_str = burst.sensing_start.strftime("%Y%m%d")
        pol = burst.polarization
        # json_path = f'{cfg.output_dir}/{burst_id}_{date_str}_{pol}.json'
        # with open(json_path, 'w') as f_json:
        #     metadata.to_file(f_json, 'json')
    '''

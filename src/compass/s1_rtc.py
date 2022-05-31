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


# TODO: remove PLAnT mosaicking and bands merging
import plant
def _mosaic(input_files, output_file):
    plant.mosaic(input_files, output_file = output_file,
                 no_average=True, force=True, interp='average')
def _merge_bands(input_files, output_file):
    plant.util(input_files, output_file = output_file, force=True)



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
    memory_mode = geocode_namespace.memory_mode
    geogrid_upsampling = geocode_namespace.geogrid_upsampling
    abs_cal_factor = geocode_namespace.abs_rad_cal
    clip_max = geocode_namespace.clip_max
    clip_min = geocode_namespace.clip_min
    # geogrids = geocode_namespace.geogrids
    flag_upsample_radar_grid = geocode_namespace.upsample_radargrid
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
    exponent = 2

    # output mosaics
    geo_filename = (f'{output_dir}/'
                    f'rtc_product.bin')
    output_imagery_dict = {}
    if flag_save_nlooks:
        out_nlooks = (f'{output_dir}/'
                      f'rtc_product_nlooks.bin')
        output_nlooks_maps_list = []
    if flag_save_rtc:
        out_rtc = (f'{output_dir}/'
                      f'rtc_product_rtc.bin')
        output_rtc_maps_list = []
    if flag_save_dem:
        out_interpolated_dem = (f'{output_dir}/'
                      f'rtc_product_interpolated_dem.bin')
        output_interpolated_dem_maps_list = []

    # iterate over sub-burts
    for burst in cfg.bursts:

        t_burst_start = time.time()

        date_str = burst.sensing_start.strftime("%Y%m%d")
        burst_id = burst.burst_id
        info_channel.log(f'processing burst: {burst_id}')
        pol = burst.polarization
        geogrid = cfg.geogrids[burst_id]

        os.makedirs(scratch_path, exist_ok=True)

        scratch_path = f'{cfg.scratch_path}/{burst_id}/{date_str}'
        os.makedirs(scratch_path, exist_ok=True)

        radar_grid = burst.as_isce3_radargrid()
        # native_doppler = burst.doppler.lut2d
        orbit = burst.orbit

        temp_slc_path = f'{scratch_path}/{burst_id}_{pol}_temp.vrt'
        burst.slc_to_vrt_file(temp_slc_path)
        rdr_burst_raster = isce3.io.Raster(temp_slc_path)

        # Generate output geocoded burst raster
        geo_burst_filename = (f'{scratch_path}/'
                              f'{burst_id}_{date_str}_{pol}.bin')

        geo_burst_raster = isce3.io.Raster(
            geo_burst_filename,
            geogrid.width, geogrid.length,
            rdr_burst_raster.num_bands, gdal.GDT_Float32,
            cfg.geocoding_params.output_format)

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
                           f'{burst_id}_{date_str}_{pol}_nlooks.bin')
            out_geo_nlooks_obj = isce3.io.Raster(
                temp_nlooks,
                geogrid.width, geogrid.length, 1,
                gdal.GDT_Float32, cfg.geocoding_params.output_format)
        else:
            temp_nlooks = None
            out_geo_nlooks_obj = None

        if flag_save_rtc:
            temp_rtc = (f'{scratch_path}/'
                           f'{burst_id}_{date_str}_{pol}_rtc_anf.bin')
            out_geo_rtc_obj = isce3.io.Raster(
                temp_rtc,
                geogrid.width, geogrid.length, 1,
                gdal.GDT_Float32, cfg.geocoding_params.output_format)
        else:
            temp_rtc = None
            out_geo_rtc_obj = None

        if flag_save_dem:
            temp_interpolated_dem = (f'{scratch_path}/'
                           f'{burst_id}_{date_str}_{pol}_interpolated_dem.bin')
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
                        out_geo_dem=out_geo_dem_obj,
                        input_rtc=None,
                        output_rtc=None,
                        dem_interp_method=dem_interp_method_enum,
                        memory_mode=memory_mode)

        del geo_burst_raster
        info_channel.log(f'file saved: {geo_burst_filename}')
        if pol not in output_imagery_dict:
            output_imagery_dict[pol] = []
        output_imagery_dict[pol].append(geo_burst_filename)

        if flag_save_nlooks:
            del out_geo_nlooks_obj
            info_channel.log(f'file saved: {temp_nlooks}')
            output_nlooks_maps_list.append(temp_nlooks)
    
        if flag_save_rtc:
            del out_geo_rtc_obj
            info_channel.log(f'file saved: {temp_rtc}')
            output_rtc_maps_list.append(temp_rtc)

        if flag_save_dem:
            del out_geo_dem_obj
            info_channel.log(f'file saved: {temp_interpolated_dem}')
            output_interpolated_dem_maps_list.append(temp_interpolated_dem)

        t_burst_end = time.time()
        info_channel.log(f'elapsed time (burst): {t_burst_end - t_burst_start}')


    mosaic_pol_list = []
    for pol in output_imagery_dict.keys():
        geo_pol_filename = (f'{output_dir}/'
                            f'rtc_product_{pol}.bin')
        imagery_list = output_imagery_dict[pol]
        _mosaic(imagery_list, geo_pol_filename)
        mosaic_pol_list.append(geo_pol_filename)
    _merge_bands(mosaic_pol_list, geo_filename)

    if flag_save_nlooks:
        _mosaic(output_nlooks_maps_list, out_nlooks)
    if flag_save_rtc:
        _mosaic(output_rtc_maps_list, out_rtc)
    if flag_save_dem:
        _mosaic(output_interpolated_dem_maps_list, out_interpolated_dem)

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
        geocode_namespace.memory_mode = isce3.geocode.GeocodeMemoryMode.SINGLE_BLOCK
    elif geocode_namespace.memory_mode == 'geogrid':
        geocode_namespace.memory_mode = isce3.geocode.GeocodeMemoryMode.BLOCKS_GEOGRID
    elif geocode_namespace.memory_mode == 'geogrid_and_radargrid':
        geocode_namespace.memory_mode = isce3.geocode.GeocodeMemoryMode.BLOCKS_GEOGRID_AND_RADARGRID
    elif geocode_namespace.memory_mode == 'auto' or (geocode_namespace.memory_mode is None):
        geocode_namespace.memory_mode = isce3.geocode.GeocodeMemoryMode.AUTO
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

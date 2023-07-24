#!/usr/bin/env python

'''wrapper to geocode metadata layers'''

import time

import h5py
import isce3
import journal
import numpy as np
from osgeo import gdal
from scipy.interpolate import InterpolatedUnivariateSpline

from compass import s1_rdr2geo
from compass.s1_rdr2geo import (file_name_los_east,
                                file_name_los_north, file_name_local_incidence,
                                file_name_layover, file_name_x,
                                file_name_y, file_name_z)
from compass.s1_cslc_qa import QualityAssuranceCSLC
from compass.utils.geo_runconfig import GeoRunConfig
from compass.utils.h5_helpers import (algorithm_metadata_to_h5group,
                                      identity_to_h5group,
                                      init_geocoded_dataset,
                                      metadata_to_h5group, DATA_PATH,
                                      ROOT_PATH)
from compass.utils.helpers import (bursts_grouping_generator, get_module_name,
                                   get_time_delta_str)
from compass.utils.yaml_argparse import YamlArgparse
from compass.utils.radar_grid import get_decimated_rdr_grd


def _fix_layover_shadow_mask(static_layers_dict, h5_root, geo_grid,
                             output_params):
    '''
    kludge correctly mask invalid pixel in geocoded layover shadow to address
    isce3::geocode::geocodeCov's inability to take in an user defined invalid
    value
    layover shadow invalid value is 127 but isce3::geocode::geocodeCov uses 0
    which conflicts with the value for non layover, non shadow pixels
    '''
    dst_ds_name = 'layover_shadow_mask'

    # find if a correctly masked dataset exists
    correctly_masked_dataset_name = ''
    for dataset_name, (enabled, _) in static_layers_dict.items():
        if enabled and dataset_name != dst_ds_name:
            correctly_masked_dataset_name = dataset_name
            break

    if correctly_masked_dataset_name:
        # get mask from correctly masked dataset
        correctly_masked_dataset_arr = \
            h5_root[f'{DATA_PATH}/{correctly_masked_dataset_name}'][()]
        mask = np.isnan(correctly_masked_dataset_arr)

        # use mask from above to correctly mask shadow layover
        # save existing to temp with mask
        layover_shadow_path = f'{DATA_PATH}/{dst_ds_name}'
        temp_arr = h5_root[layover_shadow_path][()]
        temp_arr[mask] = 127

        # delete existing and rewrite with masked data
        del h5_root[layover_shadow_path]
        _ = init_geocoded_dataset(h5_root[DATA_PATH], dst_ds_name, geo_grid,
                                  dtype=None,
                                  description=np.string_(dst_ds_name),
                                  data=temp_arr, output_cfg=output_params)


def run(cfg, burst, fetch_from_scratch=False):
    '''
    Geocode metadata layers in single HDF5

    Parameters
    ----------
    cfg: GeoRunConfig
        GeoRunConfig object with user runconfig options
    burst: Sentinel1BurstSlc
        Object containing burst parameters needed for geocoding
    fetch_from_scratch: bool
        If True grabs metadata layers from scratch dir
    '''
    module_name = get_module_name(__file__)
    info_channel = journal.info(f"{module_name}.run")
    info_channel.log(f"Starting {module_name} burst")

    # Start tracking processing time
    t_start = time.perf_counter()

    # common initializations
    dem_raster = isce3.io.Raster(cfg.dem)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid
    image_grid_doppler = isce3.core.LUT2d()
    threshold = cfg.geo2rdr_params.threshold
    iters = cfg.geo2rdr_params.numiter
    lines_per_block = cfg.geo2rdr_params.lines_per_block

    # process one burst only
    date_str = burst.sensing_start.strftime("%Y%m%d")
    burst_id = str(burst.burst_id)
    geo_grid = cfg.geogrids[burst_id]
    output_epsg = geo_grid.epsg

    radar_grid = burst.as_isce3_radargrid()
    orbit = burst.orbit

    # Initialize input/output paths
    burst_id_date_key = (burst_id, date_str)
    out_paths = cfg.output_paths[burst_id_date_key]

    input_path = out_paths.output_directory
    if fetch_from_scratch:
        input_path = out_paths.scratch_directory

    # Initialize geocode object
    geocode_obj = isce3.geocode.GeocodeFloat32()
    geocode_obj.orbit = orbit
    geocode_obj.ellipsoid = ellipsoid
    geocode_obj.doppler = image_grid_doppler
    geocode_obj.threshold_geo2rdr = threshold
    geocode_obj.numiter_geo2rdr = iters
    float_bytes = 4
    block_size = lines_per_block * geo_grid.width * float_bytes
    geocode_obj.geogrid(geo_grid.start_x, geo_grid.start_y,
                geo_grid.spacing_x, geo_grid.spacing_y,
                geo_grid.width, geo_grid.length, geo_grid.epsg)

    # Init geotransform to be set in geocoded product
    geotransform = [geo_grid.start_x, geo_grid.spacing_x, 0,
                    geo_grid.start_y, 0, geo_grid.spacing_y]

    # Dict containing which layers to geocode and their respective file names
    # key: dataset name
    # value: (bool flag if dataset is to written, raster layer name, description)
    static_layers = \
        {file_name_x: (cfg.rdr2geo_params.compute_longitude, 'x',
                       'Longitude coordinate in degrees'),
         file_name_y: (cfg.rdr2geo_params.compute_latitude, 'y',
                       'Latitude coordinate in degrees'),
         file_name_z: (cfg.rdr2geo_params.compute_height, 'z',
                       'Height in meters'),
         file_name_local_incidence: (cfg.rdr2geo_params.compute_local_incidence_angle,
                                    'local_incidence',
                                     'Local incidence angle in degrees'),
         file_name_los_east: (cfg.rdr2geo_params.compute_ground_to_sat_east,
                             'los_east',
                             'East component of unit vector of LOS from target to sensor'),
         file_name_los_north: (cfg.rdr2geo_params.compute_ground_to_sat_north,
                              'los_north',
                               'North component of unit vector of LOS from target to sensor'),
         file_name_layover: (cfg.rdr2geo_params.compute_layover_shadow_mask,
                             'layover_shadow_mask',
                             'Layover shadow mask')
         }

    out_h5 = f'{out_paths.output_directory}/static_layers_{burst_id}.h5'
    with h5py.File(out_h5, 'w') as h5_root:
        # write identity and metadata to HDF5
        root_group = h5_root[ROOT_PATH]
        metadata_to_h5group(root_group, burst, cfg, save_noise_and_cal=False,
                            save_processing_parameters=False)
        identity_to_h5group(root_group, burst, cfg, 'Static layers CSLC-S1',
                            '0.1')
        algorithm_metadata_to_h5group(root_group, is_static_layers=True)

        # Create group static_layers group under DATA_PATH for consistency with
        # CSLC product
        static_layer_data_group = h5_root.require_group(DATA_PATH)

        # Geocode designated layers
        for dataset_name, (enabled, raster_file_name,
                           description) in static_layers.items():
            if not enabled:
                continue

            # init value is invalid value for the single/float32
            dtype = np.single
            # layoverShadowMask is last option, no need to change data type
            # and interpolator afterwards
            if dataset_name == 'layover_shadow_mask':
                geocode_obj.data_interpolator = 'NEAREST'
                dtype = np.byte
                # layover shadow is a char (no NaN char, 0 represents unmasked
                # value)

            # Create dataset with x/y coords/spacing and projection
            topo_ds = init_geocoded_dataset(static_layer_data_group,
                                            dataset_name, geo_grid, dtype,
                                            description,
                                            output_cfg=cfg.output_params)

            # Init output and input isce3.io.Raster objects for geocoding
            output_raster = isce3.io.Raster(f"IH5:::ID={topo_ds.id.id}".encode("utf-8"),
                                            update=True)

            input_raster = isce3.io.Raster(f'{input_path}/{raster_file_name}.rdr')

            geocode_obj.geocode(radar_grid=radar_grid,
                                input_raster=input_raster,
                                output_raster=output_raster,
                                dem_raster=dem_raster,
                                output_mode=isce3.geocode.GeocodeOutputMode.INTERP,
                                min_block_size=block_size,
                                max_block_size=block_size)
            output_raster.set_geotransform(geotransform)
            output_raster.set_epsg(output_epsg)
            del input_raster
            del output_raster

            if dataset_name == 'layover_shadow_mask':
                _fix_layover_shadow_mask(static_layers, h5_root, geo_grid,
                                         cfg.output_params)

    if cfg.quality_assurance_params.perform_qa:
        cslc_qa = QualityAssuranceCSLC()
        with h5py.File(out_h5, 'a') as h5_root:
            cslc_qa.compute_static_layer_stats(h5_root, cfg.rdr2geo_params)
            cslc_qa.shadow_pixel_classification(h5_root)
            cslc_qa.set_orbit_type(cfg, h5_root)
            if cfg.quality_assurance_params.output_to_json:
                cslc_qa.write_qa_dicts_to_json(out_paths.stats_json_path)

    dt = get_time_delta_str(t_start)
    info_channel.log(
        f"{module_name} burst successfully ran in {dt} (hr:min:sec)")


def geocode_luts(geo_burst_h5, burst, cfg, dst_group_path, item_dict,
                 output_params, dec_factor_x_rng=20, dec_factor_y_az=5):
    '''
    Geocode the radiometric calibration parameters,
    and write them into output HDF5.

    Parameters
    ----------
    geo_burst_h5: h5py.files.File
        HDF5 object as the output product
    burst: s1reader.Sentinel1BurstSlc
        Sentinel-1 burst SLC
    cfg: GeoRunConfig
        GeoRunConfig object with user runconfig options
    dst_group_path: str
        Path in HDF5 where geocode rasters will be placed
    item_dict: dict
        Dict containing item names and values to be geocoded
    dec_factor_x_rg: int
        Decimation factor to downsample the LUT in
        x or range direction
    dec_factor_y_az: int
        Decimation factor to downsample the LUT in
        y or azimuth direction
    '''
    dem_raster = isce3.io.Raster(cfg.dem)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid
    burst_id = str(burst.burst_id)
    geo_grid = cfg.geogrids[burst_id]

    date_str = burst.sensing_start.strftime("%Y%m%d")
    burst_id_date_key = (burst_id, date_str)
    out_paths = cfg.output_paths[burst_id_date_key]

    # Common initializations
    threshold = cfg.geo2rdr_params.threshold
    iters = cfg.geo2rdr_params.numiter
    scratch_path = out_paths.scratch_directory

    # generate decimated radar and geo grids for LUT(s)
    decimated_geogrid = isce3.product.GeoGridParameters(
                            geo_grid.start_x,
                            geo_grid.start_y,
                            geo_grid.spacing_x * dec_factor_x_rng,
                            geo_grid.spacing_y * dec_factor_y_az,
                            int(np.ceil(geo_grid.width // dec_factor_x_rng)),
                            int(np.ceil(geo_grid.length // dec_factor_y_az)),
                            geo_grid.epsg)

    # initialize geocode object
    geocode_obj = isce3.geocode.GeocodeFloat32()
    geocode_obj.orbit = burst.orbit
    geocode_obj.ellipsoid = ellipsoid
    geocode_obj.doppler = isce3.core.LUT2d()
    geocode_obj.threshold_geo2rdr = threshold
    geocode_obj.numiter_geo2rdr = iters
    geocode_obj.geogrid(decimated_geogrid.start_x,
                        decimated_geogrid.start_y,
                        decimated_geogrid.spacing_x,
                        decimated_geogrid.spacing_y,
                        decimated_geogrid.width,
                        decimated_geogrid.length,
                        decimated_geogrid.epsg)
    dst_group =\
        geo_burst_h5.require_group(dst_group_path)

    gdal_envi_driver = gdal.GetDriverByName('ENVI')

    # Define the radargrid for LUT interpolation
    # The resultant radargrid will have
    # the very first and the last LUT values be included in the grid.
    radargrid_interp = get_decimated_rdr_grd(burst.as_isce3_radargrid(),
                                             dec_factor_x_rng, dec_factor_y_az)

    range_px_interp_vec = np.linspace(0, burst.width - 1, radargrid_interp.width)
    azimuth_px_interp_vec = np.linspace(0, burst.length - 1, radargrid_interp.length)

    for item_name, (rg_lut_grid, rg_lut_val,
                    az_lut_grid, az_lut_val) in item_dict.items():
        # prepare input dataset in output HDF5
        init_geocoded_dataset(dst_group,
                              item_name,
                              decimated_geogrid,
                              'float32',
                              f'geocoded {item_name}',
                              output_cfg=cfg.output_params)

        dst_dataset = geo_burst_h5[f'{dst_group_path}/{item_name}']

        # prepare output raster
        geocoded_cal_lut_raster =\
            isce3.io.Raster(
                f"IH5:::ID={dst_dataset.id.id}".encode("utf-8"), update=True)

        if az_lut_grid is not None:
            azimuth_px_interp_vec += az_lut_grid[0]

        # Get the interpolated range LUT
        param_interp_obj_rg = InterpolatedUnivariateSpline(rg_lut_grid,
                                                           rg_lut_val,
                                                           k=1)
        range_lut_interp = param_interp_obj_rg(range_px_interp_vec)

        # Get the interpolated azimuth LUT
        if az_lut_grid is None or az_lut_val is None:
            azimuth_lut_interp = np.ones(radargrid_interp.length)
        else:
            param_interp_obj_az = InterpolatedUnivariateSpline(az_lut_grid,
                                                               az_lut_val,
                                                               k=1)
            azimuth_lut_interp = param_interp_obj_az(azimuth_px_interp_vec)

        lut_arr = np.matmul(azimuth_lut_interp[..., np.newaxis],
                            range_lut_interp[np.newaxis, ...])

        lut_path = f'{scratch_path}/{item_name}_radargrid.rdr'
        lut_gdal_raster = gdal_envi_driver.Create(lut_path,
                                                  radargrid_interp.width,
                                                  radargrid_interp.length,
                                                  1, gdal.GDT_Float32)
        lut_band = lut_gdal_raster.GetRasterBand(1)
        lut_band.WriteArray(lut_arr)
        lut_band.FlushCache()
        lut_gdal_raster = None

        input_raster = isce3.io.Raster(lut_path)

        # geocode then set transform and EPSG in output raster
        geocode_obj.geocode(radar_grid=radargrid_interp,
                            input_raster=input_raster,
                            output_raster=geocoded_cal_lut_raster,
                            dem_raster=dem_raster,
                            output_mode=isce3.geocode.GeocodeOutputMode.INTERP)

        geotransform = \
            [decimated_geogrid.start_x, decimated_geogrid.spacing_x, 0,
             decimated_geogrid.start_y, 0, decimated_geogrid.spacing_y]

        geocoded_cal_lut_raster.set_geotransform(geotransform)
        geocoded_cal_lut_raster.set_epsg(epsg)

        del input_raster
        del geocoded_cal_lut_raster


def geocode_calibration_luts(geo_burst_h5, burst, cfg,
                             dec_factor_x_rng=20,
                             dec_factor_y_az=5):
    '''
    Geocode the radiometric calibration parameters,
    and write them into output HDF5.

    Parameters
    ----------
    geo_burst_h5: h5py.files.File
        HDF5 object as the output product
    burst: s1reader.Sentinel1BurstSlc
        Sentinel-1 burst SLC
    cfg: GeoRunConfig
        GeoRunConfig object with user runconfig options
    dec_factor_x_rg: int
        Decimation factor to downsample the LUT in
        x or range direction
    dec_factor_y_az: int
        Decimation factor to downsample the LUT in
        y or azimuth direction
    '''
    dst_group_path = f'{ROOT_PATH}/metadata/calibration_information'

    #[Range grid of the source in pixel,
    # range LUT value,
    # azimuth grid of the source in pixel,
    # azimuth LUT value]
    item_dict_calibration = {
        'gamma':[burst.burst_calibration.pixel,
                 burst.burst_calibration.gamma,
                 None,
                 None],
        'sigma_naught':[burst.burst_calibration.pixel,
                        burst.burst_calibration.sigma_naught,
                        None,
                        None],
        'dn':[burst.burst_calibration.pixel,
              burst.burst_calibration.dn,
              None,
              None]
        }
    geocode_luts(geo_burst_h5, burst, cfg, dst_group_path, item_dict_calibration,
                 cfg.output_params, dec_factor_x_rng, dec_factor_y_az)


def geocode_noise_luts(geo_burst_h5, burst, cfg,
                       dec_factor_x_rng=20,
                       dec_factor_y_az=5):
    '''
    Geocode the noise LUT, and write that into output HDF5.

    Parameters
    ----------
    geo_burst_h5: h5py.files.File
        HDF5 object as the output product
    burst: s1reader.Sentinel1BurstSlc
        Sentinel-1 burst SLC
    cfg: GeoRunConfig
        GeoRunConfig object with user runconfig options
    dec_factor_x_rg: int
        Decimation factor to downsample the LUT in
        x or range direction
    dec_factor_y_az: int
        Decimation factor to downsample the LUT in
        y or azimuth direction
    '''
    dst_group_path =  f'{ROOT_PATH}/metadata/noise_information'
    item_dict_noise = {'thermal_noise_lut': [burst.burst_noise.range_pixel,
                                       burst.burst_noise.range_lut,
                                       burst.burst_noise.azimuth_line,
                                       burst.burst_noise.azimuth_lut]
                                       }
    geocode_luts(geo_burst_h5, burst, cfg, dst_group_path, item_dict_noise,
                 cfg.output_params, dec_factor_x_rng, dec_factor_y_az)


if __name__ == "__main__":
    ''' run geocode metadata layers from command line'''
    parser = YamlArgparse()

    # Get a runconfig dict from command line args
    cfg = GeoRunConfig.load_from_yaml(parser.args.run_config_path,
                                      workflow_name='s1_cslc_geo')

    for _, bursts in bursts_grouping_generator(cfg.bursts):
        burst = bursts[0]

        # Generate required static layers
        if cfg.rdr2geo_params.enabled:
            s1_rdr2geo.run(cfg, save_in_scratch=True)

            # Geocode static layers if needed
            if cfg.rdr2geo_params.geocode_metadata_layers:
                run(cfg, burst, fetch_from_scratch=True)

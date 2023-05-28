#!/usr/bin/env python

'''wrapper to geocode metadata layers'''

from datetime import timedelta
import time

import h5py
import isce3
import journal
import numpy as np
from osgeo import gdal
from scipy.interpolate import InterpolatedUnivariateSpline

from compass import s1_rdr2geo
from compass.s1_cslc_qa import QualityAssuranceCSLC
from compass.utils.geo_runconfig import GeoRunConfig
from compass.utils.h5_helpers import (init_geocoded_dataset,
                                      metadata_to_h5group, DATA_PATH,
                                      METADATA_PATH,
                                      ROOT_PATH)
from compass.utils.helpers import bursts_grouping_generator, get_module_name
from compass.utils.yaml_argparse import YamlArgparse
from compass.utils.radar_grid import get_decimated_rdr_grd


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
    t_start = time.time()

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

    # Geocode list of products
    geotransform = [geo_grid.start_x, geo_grid.spacing_x, 0,
                    geo_grid.start_y, 0, geo_grid.spacing_y]

    # Get the metadata layers to compute
    meta_layers = {'x': cfg.rdr2geo_params.compute_longitude,
                   'y': cfg.rdr2geo_params.compute_latitude,
                   'z': cfg.rdr2geo_params.compute_height,
                   'incidence': cfg.rdr2geo_params.compute_incidence_angle,
                   'local_incidence': cfg.rdr2geo_params.compute_local_incidence_angle,
                   'heading': cfg.rdr2geo_params.compute_azimuth_angle,
                   'layover_shadow_mask': cfg.rdr2geo_params.compute_layover_shadow_mask
                   }

    out_h5 = f'{out_paths.output_directory}/static_layers_{burst_id}.h5'
    with h5py.File(out_h5, 'w') as h5_obj:
        # Create group static_layers group under DATA_PATH for consistency with
        # CSLC product
        static_layer_group = h5_obj.require_group(f'{DATA_PATH}/static_layers')

        # Geocode designated layers
        for layer_name, enabled in meta_layers.items():
            if not enabled:
                continue

            dtype = np.single
            # layoverShadowMask is last option, no need to change data type
            # and interpolator afterwards
            if layer_name == 'layover_shadow_mask':
                geocode_obj.data_interpolator = 'NEAREST'
                dtype = np.byte

            # Create dataset with x/y coords/spacing and projection
            topo_ds = init_geocoded_dataset(static_layer_group, layer_name,
                                            geo_grid, dtype,
                                            np.string_(layer_name))

            # Init output and input isce3.io.Raster objects for geocoding
            output_raster = isce3.io.Raster(f"IH5:::ID={topo_ds.id.id}".encode("utf-8"),
                                            update=True)

            input_raster = isce3.io.Raster(f'{input_path}/{layer_name}.rdr')

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

            # save metadata
            cslc_group = h5_obj.require_group(ROOT_PATH)
            metadata_to_h5group(cslc_group, burst, cfg)

    if cfg.quality_assurance_params.perform_qa:
        cslc_qa = QualityAssuranceCSLC()
        with h5py.File(out_h5, 'a') as h5_obj:
            cslc_qa.compute_static_layer_stats(h5_obj, cfg.rdr2geo_params)
            cslc_qa.shadow_pixel_classification(h5_obj)
            cslc_qa.set_orbit_type(cfg, h5_obj)
            if cfg.quality_assurance_params.output_to_json:
                cslc_qa.write_qa_dicts_to_json(out_paths.stats_json_path)

    dt = str(timedelta(seconds=time.time() - t_start)).split(".")[0]
    info_channel.log(
        f"{module_name} burst successfully ran in {dt} (hr:min:sec)")


def geocode_luts(geo_burst_h5, burst, cfg, dst_group_path, item_dict,
                 dec_factor_x_rng=20, dec_factor_y_az=5):
    '''
    Geocode the radiometric calibratio paremeters,
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
                              f'geocoded {item_name}')

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

        # geocode then set transfrom and EPSG in output raster
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
    Geocode the radiometric calibratio paremeters,
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
                 dec_factor_x_rng,
                 dec_factor_y_az)


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
                 dec_factor_x_rng,
                 dec_factor_y_az)


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

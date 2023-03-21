#!/usr/bin/env python

'''wrapper to geocode metadata layers'''

from datetime import timedelta
import os
import time

import h5py
import isce3
import journal
import numpy as np

from osgeo import gdal
from compass.utils.runconfig import RunConfig
from compass.utils.h5_helpers import (corrections_to_h5group,
                                      identity_to_h5group,
                                      init_geocoded_dataset,
                                      metadata_to_h5group,
                                      ROOT_PATH)
from compass.utils.helpers import get_module_name
from compass.utils.yaml_argparse import YamlArgparse


def run(cfg, burst, fetch_from_scratch=False):
    '''
    Geocode metadata layers in single HDF5

    Parameters
    ----------
    cfg: dict
        Dictionary with user runconfig options
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
    blocksize = cfg.geo2rdr_params.lines_per_block
    output_format = cfg.geocoding_params.output_format

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
    geo = isce3.geocode.GeocodeFloat32()
    geo.orbit = orbit
    geo.ellipsoid = ellipsoid
    geo.doppler = image_grid_doppler
    geo.threshold_geo2rdr = threshold
    geo.numiter_geo2rdr = iters
    geo.lines_per_block = blocksize
    geo.geogrid(geo_grid.start_x, geo_grid.start_y,
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
                   'layover_shadow_mask': cfg.rdr2geo_params.compute_layover_shadow_mask}

    out_h5 = f'{out_paths.output_directory}/topo.h5'
    shape = (geo_grid.length, geo_grid.width)
    with h5py.File(out_h5, 'w') as topo_h5:
        for layer_name, enabled in meta_layers.items():
            if not enabled:
                continue
            dtype = np.single
            # layoverShadowMask is last option, no need to change data type
            # and interpolator afterwards
            if layer_name == 'layover_shadow_mask':
                geo.data_interpolator = 'NEAREST'
                dtype = np.byte

            topo_ds = topo_h5.create_dataset(layer_name, dtype=dtype,
                                             shape=shape)
            topo_ds.attrs['description'] = np.string_(layer_name)
            output_raster = isce3.io.Raster(f"IH5:::ID={topo_ds.id.id}".encode("utf-8"),
                                            update=True)

            input_raster = isce3.io.Raster(f'{input_path}/{layer_name}.rdr')

            geo.geocode(radar_grid=radar_grid, input_raster=input_raster,
                        output_raster=output_raster, dem_raster=dem_raster,
                        output_mode=isce3.geocode.GeocodeOutputMode.INTERP)
            output_raster.set_geotransform(geotransform)
            output_raster.set_epsg(output_epsg)
            del input_raster
            del output_raster

    dt = str(timedelta(seconds=time.time() - t_start)).split(".")[0]
    info_channel.log(
        f"{module_name} burst successfully ran in {dt} (hr:min:sec)")


def geocode_calibration_luts(geo_burst_h5, burst, cfg, 
                             dec_factor=40):
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
    dec_factor: int
        Decimation factor to downsample the slant range pixels for LUT
    '''
    dem_raster = isce3.io.Raster(cfg.dem)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid
    burst_id = str(burst.burst_id)
    geo_grid = cfg.geogrids[burst_id]
    radar_grid = burst.as_isce3_radargrid()

    date_str = burst.sensing_start.strftime("%Y%m%d")
    burst_id_date_key = (burst_id, date_str)
    out_paths = cfg.output_paths[burst_id_date_key]

    # Common initializations
    threshold = cfg.geo2rdr_params.threshold
    iters = cfg.geo2rdr_params.numiter
    scratch_path = out_paths.scratch_directory

    # Designate radiometric calibration parameter to geocode
    calibration_dict = {
        'gamma':burst.burst_calibration.gamma,
        'sigma_naught':burst.burst_calibration.sigma_naught,
    }

    # define the geogrid for calbration LUT
    radargrid_calibration = radar_grid.multilook(dec_factor,
                                                 dec_factor)
    geogrid_calibration = isce3.product.GeoGridParameters(
                            geo_grid.start_x,
                            geo_grid.start_y,
                            geo_grid.spacing_x * dec_factor,
                            geo_grid.spacing_y * dec_factor,
                            geo_grid.width // dec_factor + 1,
                            geo_grid.length // dec_factor + 1,
                            geo_grid.epsg)

    geo_calibration = isce3.geocode.GeocodeFloat32()
    geo_calibration.orbit = burst.orbit
    geo_calibration.ellipsoid = ellipsoid
    geo_calibration.doppler = isce3.core.LUT2d()
    geo_calibration.threshold_geo2rdr = threshold
    geo_calibration.numiter_geo2rdr = iters
    geo_calibration.geogrid(geogrid_calibration.start_x,
                            geogrid_calibration.start_y,
                            geogrid_calibration.spacing_x,
                            geogrid_calibration.spacing_y,
                            geogrid_calibration.width,
                            geogrid_calibration.length,
                            geogrid_calibration.epsg)
    dem_raster = isce3.io.Raster(cfg.dem)
    calibration_group_path =\
        f'{root_path}/CSLC/metadata/calibration_information'
    calibration_group =\
        geo_burst_h5.require_group(calibration_group_path)

    drv_lut_radargrid = gdal.GetDriverByName('ENVI')
    for calibration_key, vec_calib in calibration_dict.items():
        init_geocoded_dataset(calibration_group,
                                calibration_key,
                                geogrid_calibration,
                                'float32',
                                f'geocoded {calibration_key}')

        calibration_dataset =\
            geo_burst_h5[f'{calibration_group_path}/{calibration_key}']

        calibration_burst_raster =\
            isce3.io.Raster(
                f"IH5:::ID={calibration_dataset.id.id}".encode("utf-8"),
                update=True)

        # prepare for the LUT in radar grid
        lut_arr = np.zeros((radargrid_calibration.length,
                            radargrid_calibration.width))
        outRaster = drv_lut_radargrid.Create(
                        f'{scratch_path}/{calibration_key}_radargrid.rdr',
                        radargrid_calibration.width,
                        radargrid_calibration.length,
                        1,
                        gdal.GDT_Float32)
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(lut_arr)
        outband.FlushCache()
        outRaster = None

        input_raster =\
                isce3.io.Raster(f'{scratch_path}/'
                                f'{calibration_key}_radargrid.rdr')

        geo_calibration.geocode(radar_grid=radargrid_calibration,
            input_raster=input_raster,
            output_raster=calibration_burst_raster,
            dem_raster=dem_raster,
            output_mode=isce3.geocode.GeocodeOutputMode.INTERP)
        geotransform = [geogrid_calibration.start_x,
                        geogrid_calibration.spacing_x,
                        0,
                        geogrid_calibration.start_y,
                        0,
                        geogrid_calibration.spacing_y]
        calibration_burst_raster.set_geotransform(geotransform)
        calibration_burst_raster.set_epsg(epsg)

        del input_raster
        del calibration_burst_raster


if __name__ == "__main__":
    ''' run geocode metadata layers from command line'''
    parser = YamlArgparse()

    # Get a runconfig dict from command line args
    cfg = RunConfig.load_from_yaml(parser.args.run_config_path,
                                   workflow_name='s1_cslc_radar')
    # run geocode metadata layers
    run(cfg)

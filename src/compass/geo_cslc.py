import os
import time

import isce3
import journal
import numpy as np
from osgeo import gdal
from shapely.geometry import MultiPoint

from compass.utils.geo_runconfig import GeoRunConfig
from compass.utils.range_split_spectrum import range_split_spectrum
from compass.utils.yaml_argparse import YamlArgparse


def run(cfg):
    '''
    Run geocode burst workflow with user-defined
    args stored in dictionary runconfig *cfg*

    Parameter:
    ---------
    cfg: dict
       Dictionary with user runconfig options
    '''
    info_channel = journal.info("geo_cslc.run")

    # Start tracking processing time
    t_start = time.time()
    info_channel.log("Starting geocode burst")

    # Common initializations
    dem_raster = isce3.io.Raster(cfg.dem)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid
    image_grid_doppler = isce3.core.LUT2d()
    threshold = cfg.geo2rdr_params.threshold
    iters = cfg.geo2rdr_params.numiter
    blocksize = cfg.geo2rdr_params.lines_per_block
    dem_margin = cfg.geocoding_params.dem_margin
    flatten = cfg.geocoding_params.flatten

    for bursts in cfg.bursts:
        burst_id = bursts[0].burst_id
        geo_grid = cfg.geogrids[burst_id]

        output_path = f'{cfg.product_path}/{burst_id}'
        scratch_path = f'{cfg.scratch_path}/{burst_id}'
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(scratch_path, exist_ok=True)
        for burst in bursts:
            radar_grid = burst.as_isce3_radargrid()
            native_doppler = burst.doppler.lut2d
            orbit = burst.orbit

            # Get azimuth polynomial coefficients for this burst
            az_carrier_poly2d = burst.get_az_carrier_poly()

            # Split the range bandwidth of the burst, if required
            if cfg.split_spectrum_params.enabled:
                rdr_burst_raster = range_split_spectrum(burst,
                                                        cfg.split_spectrum_params,
                                                        scratch_path)
            else:
                temp_slc_path = f'{scratch_path}/{burst.burst_id}_temp.tiff'
                burst.slc_to_file(temp_slc_path, 'GTiff')
                rdr_burst_raster = isce3.io.Raster(temp_slc_path)

            # Generate output geocoded burst raster
            geo_burst_raster = isce3.io.Raster(
                f'{output_path}/geo_{burst.burst_id}',
                geo_grid.width, geo_grid.length,
                rdr_burst_raster.num_bands, gdal.GDT_CFloat32,
                'ENVI')

            # Extract burst boundaries
            b_bounds = np.s_[burst.first_valid_line:burst.last_valid_line,
                             burst.first_valid_sample:burst.last_valid_sample]

            # Create sliced radar grid from burst boundaries
            sliced_radar_grid = burst.as_isce3_radargrid()[b_bounds]

            # Geocode
            isce3.geocode.geocode_slc(geo_burst_raster, rdr_burst_raster,
                                      dem_raster,
                                      radar_grid, sliced_radar_grid,
                                      geo_grid, orbit,
                                      native_doppler,
                                      image_grid_doppler, ellipsoid, threshold,
                                      iters,
                                      blocksize, dem_margin, flatten,
                                      azimuth_carrier=az_carrier_poly2d)

            # Set geo transformation
            geotransform = [geo_grid.start_x, geo_grid.spacing_x, 0,
                            geo_grid.start_y, 0, geo_grid.spacing_y]
            geo_burst_raster.set_geotransform(geotransform)
            geo_burst_raster.set_epsg(epsg)
            del geo_burst_raster

            # Get polygon including valid areas (to be dumped in metadata)
            filename = f'{output_path}/geo_{burst.burst_id}'
            poly = get_valid_polygon(filename, np.nan)

    dt = time.time() - t_start
    info_channel.log(f'geocode burst successfully ran in {dt:.3f} seconds')


def pixel2coords(ds, xpix, ypix):
    '''
    Get coordinates of pixel location (x_pix, y_pix)

    Parameters:
    ----------
    ds: gdal.Open
       GDAL dataset handle
    xpix: int
       Location of pixel along columns
    ypix: int
       Location of pixel along rows

    Returns:
    -------
    px: float
       X coordinates corresponding to xpix
    py: float
       Y coordinates corresponding to ypix
    '''
    geo_transf = ds.GetGeoTransform()
    xmin = geo_transf[0]
    xsize = geo_transf[1]
    ymin = geo_transf[3]
    ysize = geo_transf[5]

    px = xpix * xsize + xmin
    py = ypix * ysize + ymin

    return (px, py)


def get_valid_polygon(filename, invalid_value):
    '''
    Get boundary polygon for raster in 'filename'.
     Polygon includes only valid pixels

    Parameters:
    ----------
    filename: str
        File path where raster is stored
    invalid_value: np.nan or float
        Invalid data value for raster in 'filename'

    Returns:
    --------
    poly: shapely.Polygon
        Shapely polygon including valid values
    '''
    # Optimize this with block-processing?
    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    burst = ds.GetRasterBand(1).ReadAsArray()

    if np.isnan(invalid_value):
        idy, idx = np.where((~np.isnan(burst.real)) &
                            (~np.isnan(burst.imag)))
    else:
        idy, idx = np.where((burst.real == invalid_value) &
                            (burst.imag == invalid_value))
    tgt_x = []
    tgt_y = []

    for x_idx, y_idy in zip(idx[::100], idy[::100]):
        px, py = pixel2coords(ds, x_idx, y_idy)
        tgt_x.append(px)
        tgt_y.append(py)

    points = MultiPoint(list(zip(tgt_x, tgt_y)))
    poly = points.convex_hull
    return poly


if __name__ == "__main__":
    '''Run geocode cslc workflow from command line'''
    # load arguments from command line
    geo_parser = YamlArgparse()

    # Get a runconfig dict from command line argumens
    runconfig = GeoRunConfig.load_from_yaml(geo_parser.run_config_path,
                                            'geo_cslc_s1')

    # Save metadata for stitching
    json_path = f'{runconfig.product_path}/metadata.json'
    with open(json_path, 'w') as f_json:
        runconfig.to_json(f_json)

    # Run geocode burst workflow
    run(runconfig)

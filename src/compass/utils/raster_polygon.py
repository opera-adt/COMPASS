'''
Functions for computing and adding boundary polygon to geo_runconfig dicts
'''
from datetime import datetime
import os

import numpy as np
from osgeo import gdal
from shapely.geometry import MultiPoint

from compass.utils import helpers


def add_poly_to_dict(geo_raster_path, burst_dict):
    '''Add boundary polygon string for associated geocoded raster

    Parameters
    ----------
    geo_raster_path: str
        Path to geocoded raster
    burst_dict: dict
        Burst dict where poly is to be added
    '''
    if not os.path.isfile(geo_raster_path):
        raise FileNotFoundError('cannont generate raster boundary - '
                                f'{geo_raster_path} not found')

    # Get polygon including valid areas (to be dumped in metadata)
    poly = get_boundary_polygon(geo_raster_path, np.nan)

    # Add polygon to dict
    burst_dict['poly'] = str(poly.wkt)


def get_boundary_polygon(filename, invalid_value):
    '''
    Get boundary polygon for raster in 'filename'.
     Polygon includes only valid pixels

    Parameters
    ----------
    filename: str
        Path to GDAL friendly file containing geocoded raster
    invalid_value: np.nan or float
        Invalid data value for raster in 'filename'

    Returns
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

    # Get geotransform defining geogrid
    xmin, xsize, _, ymin, _, ysize = ds.GetGeoTransform()

    # Use geotransform to convert indices to geogrid points
    tgt_xy = [[x_idx * xsize + xmin, y_idx * ysize + ymin]
              for x_idx, y_idx in zip(idx[::100], idy[::100])]

    points = MultiPoint(tgt_xy)
    poly = points.convex_hull
    return poly

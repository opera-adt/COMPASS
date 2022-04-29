'''
Functions for computing and adding boundary polygon to geo_runconfig dicts
'''
import os

import numpy as np
from osgeo import gdal
from shapely.geometry import MultiPoint


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
    if not os.path.isfile(filename):
        raise FileNotFoundError('cannont generate raster boundary - '
                                f'{filename} not found')

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

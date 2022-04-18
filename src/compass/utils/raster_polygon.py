'''
Functions for computing and adding boundary polygon to geo_runconfig dicts
'''
from datetime import datetime

import numpy as np
from osgeo import gdal
from shapely.geometry import MultiPoint

from compass.utils import helpers


def add_poly_to_dict(top_output_dir, burst_dict):
    '''Add boundary polygon string for associated geocoded raster

    Parameters
    ----------
    top_output_dir: str
        Path to parent directory where geocoded products are saved. Expected
        directory structure:
            top_output_dir
            └── <burst date MMDDYYYY>
                └── geo_<burst_id>_<pol>
    burst_dict: dict
        Burst dict where poly is to be added
    '''
    validate_burst_dict(burst_dict)

    burst_id = burst_dict['burst_id']
    date_str = datetime(burst_dict['sensing_start']).date()
    pol = burst_dict['polarization']

    # Set output path and check if it exists
    burst_output_dir = f'{top_output_dir}/{date_str}/{burst_id}_{pol}'
    helpers.check_file_path(burst_output_dir)

    # Get polygon including valid areas (to be dumped in metadata)
    filename = f'{burst_output_dir}/geo_{burst_id}_{pol}'
    poly = get_boundary_polygon(filename, np.nan)

    # Add polygon to dict
    burst_dict['poly'] = poly.wkt


def validate_burst_dict(burst_dict):
    '''Validate requisite values in burst dict

    burst_dict: dict
        Burst dict to be validated
    '''
    # Make sure requisite keys present
    for key in ['burst_id', 'sensing_start', 'polarization']:
        if key not in burst_dict:
            raise KeyError(f'{key} not found in burst_dict')

    # Check date format is correct
    try:
        _ = datetime(burst_dict['sensing_start'])
    except ValueError:
        print(f'{burst_dict["sensing_start"]} is not correctly formatted for datetime conversion')

    # Make sure polarization is valid
    valid_pols = ['vv', 'vh', 'hh', 'hv']
    burst_pol = burst_dict['polarization']
    if burst_pol not in valid_pols:
        raise ValueError(f'{burst_pol} not in valid polarizations {valid_pols}')


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

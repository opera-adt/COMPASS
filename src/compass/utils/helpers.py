'''collection of useful functions used across workflows'''

import itertools
import os
import sqlite3

import isce3
import journal
import numpy as np
from pyproj.transformer import Transformer
from osgeo import gdal
from shapely import geometry

import compass


WORKFLOW_SCRIPTS_DIR = os.path.dirname(compass.__file__)

# get the basename given an input file path
# example: get_module_name(__file__)
get_module_name = lambda x : os.path.basename(x).split('.')[0]


def check_file_path(file_path: str) -> None:
    """Check if file_path exist else raise an error.

    Parameters
    ----------
    file_path : str
        Path to file to be checked
    """
    error_channel = journal.error('helpers.check_file_path')
    if not os.path.exists(file_path):
        err_str = f'{file_path} not found'
        error_channel.log(err_str)
        raise FileNotFoundError(err_str)


def check_directory(file_path: str) -> None:
    """Check if directory in file_path exists else raise an error.

    Parameters
    ----------
    file_path: str
       Path to directory to be checked
    """
    error_channel = journal.error('helpers.check_directory')
    if not os.path.isdir(file_path):
        err_str = f'{file_path} not found'
        error_channel.log(err_str)
        raise FileNotFoundError(err_str)


def get_file_polarization_mode(file_path: str) -> str:
    '''Check polarization mode from file name

    Taking PP from SAFE file name with following format:
    MMM_BB_TTTR_LFPP_YYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS_OOOOOO_DDDDDD_CCCC.SAFE

    Parameters
    ----------
    file_path : str
        SAFE file name to parse

    Returns
    -------
    original: dict
        Default dictionary updated with user-defined options

    References
    ----------
    https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/naming-conventions
    '''
    # index split tokens from rear to account for R in TTTR being possibly
    # replaced with '_'
    safe_pol_mode = os.path.basename(file_path).split('_')[-6][2:]

    return safe_pol_mode


def deep_update(original, update):
    """Update default runconfig dict with user-supplied dict.

    Parameters
    ----------
    original : dict
        Dict with default options to be updated
    update: dict
        Dict with user-defined options used to update original/default

    Returns
    -------
    original: dict
        Default dictionary updated with user-defined options

    References
    ----------
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for key, val in update.items():
        if isinstance(val, dict):
            original[key] = deep_update(original.get(key, {}), val)
        else:
            original[key] = val

    # return updated original
    return original


def check_write_dir(dst_path: str):
    """Check if given directory is writeable; else raise error.

    Parameters
    ----------
    dst_path : str
        File path to directory for which to check writing permission
    """
    if not dst_path:
        dst_path = '.'

    error_channel = journal.error('helpers.check_write_dir')

    # check if scratch path exists
    dst_path_ok = os.path.isdir(dst_path)

    if not dst_path_ok:
        try:
            os.makedirs(dst_path, exist_ok=True)
        except OSError:
            err_str = f"Unable to create {dst_path}"
            error_channel.log(err_str)
            raise OSError(err_str)

    # check if path writeable
    write_ok = os.access(dst_path, os.W_OK)
    if not write_ok:
        err_str = f"{dst_path} scratch directory lacks write permission."
        error_channel.log(err_str)
        raise PermissionError(err_str)


def check_dem(dem_path: str):
    """Check if given path is a GDAL-compatible file; else raise error

    Parameters
    ----------
    dem_path : str
        File path to DEM for which to check GDAL-compatibility
    """
    error_channel = journal.error('helpers.check_dem')
    try:
        gdal.Open(dem_path, gdal.GA_ReadOnly)
    except ValueError:
        err_str = f'{dem_path} cannot be opened by GDAL'
        error_channel.log(err_str)
        raise ValueError(err_str)

    epsg = isce3.io.Raster(dem_path).get_epsg()
    if not 1024 <= epsg <= 32767:
        err_str = f'DEM epsg of {epsg} out of bounds'
        error_channel.log(err_str)
        raise ValueError(err_str)


def bbox_to_utm(bbox, *, epsg_src, epsg_dst):
    """Convert bounding box coordinates to UTM.

    Parameters
    ----------
    bbox : tuple
        Tuple containing the lon/lat bounding box coordinates
        (left, bottom, right, top) in degrees
    epsg_src : int
        EPSG code identifying input bbox coordinate system
    epsg_dst : int
        EPSG code identifying output coordinate system

    Returns
    -------
    tuple
        Tuple containing the bounding box coordinates in UTM (meters)
        (left, bottom, right, top)
    """
    xmin, ymin, xmax, ymax = bbox
    xys = _convert_to_utm([(xmin, ymin), (xmax, ymax)], epsg_src, epsg_dst)
    return (*xys[0], *xys[1])


def polygon_to_utm(poly, *, epsg_src, epsg_dst):
    """Convert a shapely.Polygon's coordinates to UTM.

    Parameters
    ----------
    poly: shapely.geometry.Polygon
        Polygon object
    epsg : int
        EPSG code identifying output projection system

    Returns
    -------
    tuple
        Tuple containing the bounding box coordinates in UTM (meters)
        (left, bottom, right, top)
    """
    coords = np.array(poly.exterior.coords)
    xys = _convert_to_utm(coords, epsg_src, epsg_dst)
    return geometry.Polygon(xys)


def _convert_to_utm(points_xy, epsg_src, epsg_dst):
    """Convert a list of points to a specified UTM coordinate system.

    If epsg_src is 4326 (lat/lon), assumes points_xy are in degrees.
    """
    if epsg_dst == epsg_src:
        return points_xy

    t = Transformer.from_crs(epsg_src, epsg_dst, always_xy=True)
    xs, ys = np.array(points_xy).T
    xt, yt = t.transform(xs, ys)
    return list(zip(xt, yt))


def burst_bbox_from_db(burst_id, burst_db_file=None, burst_db_conn=None):
    """Find the bounding box of a burst in the database.

    Parameters
    ----------
    burst_id : str
        JPL burst ID
    burst_db_file : str
        Location of burst database sqlite file, by default None
    burst_db_conn : sqlite3.Connection
        Connection object to burst database (If already connected)
        Alternative to providing burst_db_file, will be faster
        for multiply queries.

    Returns
    -------
    epsg : int
        EPSG code(s) of burst bounding box(es)
    bbox : tuple[float]
        Bounding box of burst in EPSG coordinates. Bounding box given as
        tuple(xmin, ymin, xmax, ymax)

    Raises
    ------
    ValueError
        If burst_id is not found in burst database
    """
    # example burst db:
    # /home/staniewi/dev/burst_map_IW_000001_375887.OPERA-JPL.sqlite3
    if burst_db_conn is None:
        burst_db_conn = sqlite3.connect(burst_db_file)
    burst_db_conn.row_factory = sqlite3.Row  # return rows as dicts

    query = "SELECT epsg, xmin, ymin, xmax, ymax FROM burst_id_map WHERE burst_id_jpl = ?"
    cur = burst_db_conn.execute(query, (burst_id,))
    result = cur.fetchone()

    if not result:
        raise ValueError(f"Failed to find {burst_id} in {burst_db_file}")

    epsg = result["epsg"]
    bbox = (result["xmin"], result["ymin"], result["xmax"], result["ymax"])

    return epsg, bbox


def burst_bboxes_from_db(burst_ids, burst_db_file=None, burst_db_conn=None):
    """Find the bounding box of bursts in the database.

    Parameters
    ----------
    burst_id : list[str]
        list of JPL burst IDs.
    burst_db_file : str
        Location of burst database sqlite file, by default None
    burst_db_conn : sqlite3.Connection
        Connection object to burst database (If already connected)
        Alternative to providing burst_db_file, will be faster
        for multiply queries.

    Returns
    -------
    bboxes : dict
        Burst bounding boxes as a dict with burst IDs as key and tuples of
        EPSG and bounding boxes (tuple[float]) as values. Bounding box given as
        tuple(xmin, ymin, xmax, ymax)

    Raises
    ------
    ValueError
        If no burst_ids are found in burst database
    """
    # example burst db:
    # /home/staniewi/dev/burst_map_IW_000001_375887.OPERA-JPL.sqlite3
    if burst_db_conn is None:
        burst_db_conn = sqlite3.connect(burst_db_file)
    burst_db_conn.row_factory = sqlite3.Row  # return rows as dicts

    # concatenate '?, ' with for each burst ID for IN query
    qs_in_query = ', '.join('?' for _ in burst_ids)
    query = f"SELECT * FROM burst_id_map WHERE burst_id_jpl IN ({qs_in_query})"
    cur = burst_db_conn.execute(query, burst_ids)
    results = cur.fetchall()

    if not results:
        raise ValueError(f"Failed to find {burst_ids} in {burst_db_file}")

    n_results = len(results)
    epsgs = [[]] * n_results
    bboxes = [[]] * n_results
    burst_ids = [[]] * n_results
    for i_result, result in enumerate(results):
        epsgs[i_result] = result["epsg"]
        bboxes[i_result] = (result["xmin"], result["ymin"],
                           result["xmax"], result["ymax"])
        burst_ids[i_result] = result["burst_id_jpl"]

    # TODO add warning if not all burst bounding boxes found
    return dict(zip(burst_ids, zip(epsgs, bboxes)))


def open_raster(filename, band=1):
    '''
    Return band as numpy array from gdal-friendly raster

    Parameters
    ----------
    filename: str
        Path where is stored GDAL raster to open
    band: int
        Band number to open

    Returns
    -------
    raster: np.ndarray
        Numpy array containing the raster band to open
    '''
    try:
        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        raster = ds.GetRasterBand(band).ReadAsArray()
    except ValueError:
        error_channel = journal.error('helpers.check_dem')
        err_str = f'{filename} cannot be opened by GDAL'
        error_channel.log(err_str)
        raise ValueError(err_str)

    return raster


def write_raster(filename, data_list, descriptions,
                 data_type=gdal.GDT_Float32, data_format='ENVI'):
    '''
    Write a multiband GDAL-friendly raster to disk.
    Each dataset allocated in the output file contains
    a description of the dataset allocated for that band

    Parameters
    ----------
    filename: str
        File path where to store output dataset
    data_list: list[np.ndarray]
        List of numpy.ndarray to allocate for each
        raster band. All datasets within the list
        are assumed to have the same shape
    descriptions: list[str]
        List of strings containing a description
        for the bands to allocate
    data_type: gdal.dtype
        GDAL dataset type
    format: gdal.Format
        Format for GDAL output file
    '''

    error_channel = journal.error('helpers.write_raster')

    # Check number of datasets match number of descriptions
    if len(data_list) != len(descriptions):
        err_str = f'Number of datasets to write does not match' \
                  f'the number of descriptions ' \
                  f'{len(data_list)} != {len(descriptions)}'
        error_channel.log(err_str)
        raise ValueError(err_str)

    # Get the shape of a dataset within the list. All the datasets
    # are assumed to have the same shape
    length, width = data_list[0].shape
    nbands = len(data_list)

    driver = gdal.GetDriverByName(data_format)
    out_ds = driver.Create(filename, width, length, nbands, data_type)

    band = 0
    for data, description in zip(data_list, descriptions):
        band += 1
        raster_band = out_ds.GetRasterBand(band)
        raster_band.SetDescription(description)
        raster_band.WriteArray(data)

    out_ds.FlushCache()


def bursts_grouping_generator(bursts):
    '''
    Dict to group bursts with the same burst ID but different polarizations
    key: burst ID, value: list[S1BurstSlc]

    Parameters
    ----------
    bursts: list[Sentinel1BurstSlc]
        List of bursts to grouped

    Yields
    ------
    k: S1BurstId
        Burst ID of grouped list of bursts
    v: list[Sentinel1BurstSlc]
        List of bursts with the same burst ID
    '''
    grouped_bursts = itertools.groupby(bursts, key=lambda b: str(b.burst_id))

    for k, v in grouped_bursts:
        yield k, list(v)

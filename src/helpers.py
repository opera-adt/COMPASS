'''
Suite of functionalities used across the CSLC workflow
'''

import logging
import os

from osgeo import gdal

WORKFLOW_SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))


def check_file_path(file_path: str) -> None:
    """Check if file_path exist. If not, raise an error
       Parameters
       ----------
       file_path: str
         Path to file to be checked
    """
    if not os.path.isfile(file_path):
        err_str = f'{file_path} not found'
        logging.error(err_str)
        raise FileNotFoundError(err_str)


def deep_update(original: dict, update: dict):
    """Update default runconfig with user-supplied options
       Parameters
       ----------
       original: dict
         Dictionary with default options to be updated
       update: dict
         Dictionary with user-supplied options used as update

       Returns
       -------
       original: dict
         "original" dictionary updated with user options

       Reference
       ---------
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
    """Check if dst_path exists and is writeable. Raise error if not.
       Parameters
       ----------
       dst_path: str
          File path to directory to be checked
    """
    if not dst_path:
        dst_path = '.'

    # Check if dst_path exists
    dst_path_ok = os.path.isdir(dst_path)

    if not dst_path_ok:
        try:
            os.makedirs(dst_path, exist_ok=True)
        except OSError:
            err_str = f"Unable to create directory {dst_path}"
            logging.error(err_str)
            raise OSError(err_str)

    # check if path writeable
    write_ok = os.access(dst_path, os.W_OK)
    if not write_ok:
        err_str = f"{dst_path} scratch directory lacks write permission."
        logging.error(err_str)
        raise PermissionError(err_str)


def check_dem(dem_path: str):
    """Check if DEM exists. DEM must be a GDAL-compatible raster
       Parameters
       ----------
       dem_path: str
         File path to DEM file to be checked
    """
    try:
        gdal.Open(dem_path)
    except ValueError:
        err_str = f'{dem_path} cannot be opened by GDAL'
        logging.error(err_str)
        raise ValueError(err_str)

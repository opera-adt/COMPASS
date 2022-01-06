'''
collection of useful functions used across workflows
'''

import journal
import os

from osgeo import gdal

def check_file_path(file_path: str) -> None:
    """Check if file_path exists. If not, raise an error.

       Parameters
       ----------
       file_path : str
           Path to file to be checked
    """
    error_channel = journal.error('helpers.check_file_path')
    if not os.path.isfile(file_path):
        err_str = f'{file_path} not found'
        error_channel.log(err_str)
        raise FileNotFoundError(err_str)

def deep_update(original, update):
    """Update default runconfig ('original') with user-supplied
       dictionary ('update').

       Parameters
       ----------
       original : dict
          Dictionary with default options to be updated
       update: dict
          Dictionary with user-defined options to use in update

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

WORKFLOW_SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))

def check_write_dir(dst_path: str):
    """Check if directory in 'dst_path' is writeable.
       If not, raise an error.

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
    """Check if DEM in 'dem_path' is a GDAL-compatible file.
       If not, raise an error

       Parameters
       ----------
       dem_path : str
          File path to DEM for which to check GDAL-compatibility
    """
    error_channel = journal.error('helpers.check_dem')
    try:
        gdal.Open(dem_path)
    except:
        err_str = f'{dem_path} cannot be opened by GDAL'
        error_channel.log(err_str)
        raise ValueError(err_str)

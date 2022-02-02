'''collection of useful functions used across workflows'''

import os

import journal
from osgeo import gdal

WORKFLOW_SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))


def check_file_path(file_path: str) -> None:
    """Check if file_path and polarizations exist else raise an error.

    Parameters
    ----------
    file_path : str
        Path to file to be checked
    polarizations : list[str]
        Path to file to be checked
    """
    error_channel = journal.error('helpers.check_file_path')
    if not os.path.isfile(file_path):
        err_str = f'{file_path} not found'
        error_channel.log(err_str)
        raise FileNotFoundError(err_str)


def get_file_polarization_mode(file_path: str) -> str:
    '''Get polarization mode from file name

    File name parsed according to:
    https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/naming-conventions

    Parameters
    ----------
    file_path : str
        File name to parse

    Returns
    -------
    _ : str
        Polarization mode (SH, SV, DH, DV)
    '''
    return os.path.basename(file_path).split('_')[-6][2:]


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
    except:
        err_str = f'{dem_path} cannot be opened by GDAL'
        error_channel.log(err_str)
        raise ValueError(err_str)

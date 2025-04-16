# Set of functionalities to download and read TEC maps
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Jun 2022                           #
############################################################
# Links:
#   IGS (NASA): https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/atmospheric_products.html
#   IMPC (DLR): https://impc.dlr.de/products/total-electron-content/near-real-time-tec/near-real-time-tec-maps-global

import datetime as dt
import logging
import os
import re

import isce3
import numpy as np
from scipy import interpolate


def read_ionex(tec_file):
    """
    Read Total Electron Content (TEC) file in
    IONEX format

    Parameters
    ----------
    tec_file: str
        Path to the TEC file in IONEX format

    Returns
    -------
    mins: np.ndarray
        1D array with time of the day in minutes
    lats: np.ndarray
        1D array with latitude in degrees
    lons: np.ndarray
        1D array with longitude in degrees
    tec_maps: np.ndarray
        3D array with vertical TEC in TECU
    rms_maps: np.ndarray
        3d array with vertical TEC RMS in TECU
    """

    # functions for parsing strings from ionex file
    # link: https://github.com/daniestevez/jupyter_notebooks/blob/master/IONEX.ipynb
    def parse_map(tec_map_str, key="TEC", exponent=-1):
        tec_map_str = re.split(f".*END OF {key} MAP", tec_map_str)[0]
        tec_map = [
            np.fromstring(x, sep=" ")
            for x in re.split(".*LAT/LON1/LON2/DLON/H\\n", tec_map_str)[1:]
        ]
        return np.stack(tec_map) * 10**exponent

    # read IONEX file
    with open(tec_file) as f:
        fc = f.read()

        # read header
        header = fc.split("END OF HEADER")[0].split("\n")
        for line in header:
            if line.strip().endswith("# OF MAPS IN FILE"):
                num_map = int(line.split()[0])
            elif line.strip().endswith("DLAT"):
                lat0, lat1, lat_step = (float(x) for x in line.split()[:3])
            elif line.strip().endswith("DLON"):
                lon0, lon1, lon_step = (float(x) for x in line.split()[:3])
            elif line.strip().endswith("EXPONENT"):
                exponent = float(line.split()[0])

        # spatial coordinates
        num_lat = (lat1 - lat0) // lat_step + 1
        num_lon = (lon1 - lon0) // lon_step + 1
        lats = np.arange(lat0, lat0 + num_lat * lat_step, lat_step)
        lons = np.arange(lon0, lon0 + num_lon * lon_step, lon_step)

        # time stamps in minutes
        min_step = 24 * 60 / (num_map - 1)
        mins = np.arange(0, num_map * min_step, min_step)

        # read TEC and its RMS maps
        tec_maps = np.array(
            [
                parse_map(t, key="TEC", exponent=exponent)
                for t in fc.split("START OF TEC MAP")[1:]
            ],
            dtype=np.float32,
        )
        rms_maps = np.array(
            [
                parse_map(t, key="RMS", exponent=exponent)
                for t in fc.split("START OF RMS MAP")[1:]
            ],
            dtype=np.float32,
        )

    return mins, lats, lons, tec_maps, rms_maps


def get_ionex_value(
    tec_file, utc_sec, lat, lon, interp_method="linear3d", rotate_tec_map=False
):
    """
    Get the TEC value from input IONEX file for the input lat/lon/datetime.
    Reference:
        Schaer, S., Gurtner, W., & Feltens, J. (1998). IONEX: The ionosphere map exchange format
        version 1.1. Paper presented at the Proceedings of the IGS AC workshop, Darmstadt, Germany.
    Parameters
    ----------
    tec_file: str
        Path of local TEC file
    utc_sec: float or 1D np.ndarray
        UTC time of the day in seconds
    lat: float or 1D np.ndarray
        Latitude in degrees
    lon: float or 1D np.ndarray
        Longitude in degrees
    interp_method: str
       Interpolation method (nearest, linear, linear2d, bilinear, linear3d, trilinear)
    rotate_tec_map: bool
        Rotate the TEC map along the SUN direction
        (for "interp_method = linear3d" only)

    Returns
    -------
    tec_val: float or 1D np.ndarray
        Vertical TEC value in TECU
    """

    def interp_3d_rotate(interpfs, mins, lats, lons, utc_min, lat, lon):
        ind0 = np.where((mins - utc_min) <= 0)[0][-1]
        ind1 = ind0 + 1
        lon0 = lon + (utc_min - mins[ind0]) * 360.0 / (24.0 * 60.0)
        lon1 = lon + (utc_min - mins[ind1]) * 360.0 / (24.0 * 60.0)
        tec_val0 = interpfs[ind0](lon0, lat)
        tec_val1 = interpfs[ind1](lon1, lat)
        tec_val = (mins[ind1] - utc_min) / (mins[ind1] - mins[ind0]) * tec_val0 + (
            utc_min - mins[ind0]
        ) / (mins[ind1] - mins[ind0]) * tec_val1
        return tec_val

    # time info
    utc_min = utc_sec / 60.0

    # read TEC file
    mins, lats, lons, tec_maps = read_ionex(tec_file)[:4]

    # resample
    if interp_method == "nearest":
        lon_ind = np.abs(lons - lon).argmin()
        lat_ind = np.abs(lats - lat).argmin()
        time_ind = np.abs(mins - utc_min).argmin()
        tec_val = tec_maps[time_ind, lat_ind, lon_ind]

    elif interp_method in ["linear", "linear2d", "bilinear"]:
        time_ind = np.abs(mins.reshape(-1, 1) - utc_min).argmin(axis=0)

        if isinstance(utc_min, np.ndarray):
            num_pts = len(utc_min)
            tec_val = np.zeros(num_pts, dtype=np.float32)
            for i in range(num_pts):
                tec_val[i] = interpolate.interp2d(
                    x=lons,
                    y=lats,
                    z=tec_maps[time_ind[i], :, :],
                    kind="linear",
                )(lon[i], lat[i])
        else:
            tec_val = interpolate.interp2d(
                x=lons,
                y=lats,
                z=tec_maps[time_ind[0], :, :],
                kind="linear",
            )(lon, lat)
    elif interp_method in ["linear3d", "trilinear"]:
        if not rotate_tec_map:
            # option 1: interpolate between consecutive TEC maps
            # testings shows better agreement with SAR obs than option 2.
            tec_val = interpolate.interpn(
                points=(mins, np.ascontiguousarray(np.flip(lats)), lons),
                values=np.flip(tec_maps, axis=1),
                xi=(utc_min, lat, lon),
                method="linear",
            )
        else:
            # option 2: interpolate between consecutive rotated TEC maps
            # reference: equation (3) in Schaer and Gurtner (1998)

            # prepare interpolation functions in advance to speed up
            interpfs = []
            for i in range(len(mins)):
                interpfs.append(
                    interpolate.interp2d(
                        x=lons,
                        y=lats,
                        z=tec_maps[i, :, :],
                        kind="linear",
                    ),
                )

            if isinstance(utc_min, np.ndarray):
                num_pts = len(utc_min)
                tec_val = np.zeros(num_pts, dtype=np.float32)
                for i in range(num_pts):
                    tec_val[i] = interp_3d_rotate(
                        interpfs,
                        mins,
                        lats,
                        lons,
                        utc_min[i],
                        lat[i],
                        lon[i],
                    )
            else:
                tec_val = interp_3d_rotate(
                    interpfs,
                    mins,
                    lats,
                    lons,
                    utc_min,
                    lat,
                    lon,
                )

    else:
        msg = f"Un-recognized interp_method input: {interp_method}!"
        msg += "\nSupported inputs: nearest, linear2d, linear3d."
        logging.error(msg)
        raise ValueError(msg)

    return tec_val


def download_ionex(date_str, tec_dir, sol_code="jpl", date_fmt="%Y%m%d"):
    """
    Download IGS vertical TEC files in IONEX format

    Parameters
    ----------
    date_str: str
        Date in date_fmt format
    tec_dir: str
        Local directory where to save downloaded files
    sol_code: str
        IGS TEC analysis center code
    date_fmt: str
        Date format code

    Returns
    -------
    fname_dst_uncomp: str
        Path to local uncompressed IONEX file
    """
    # get the source (remote) and destination (local) file path/url
    kwargs = dict(sol_code=sol_code, date_fmt=date_fmt)
    fname_src = get_ionex_filename(date_str, tec_dir=None, **kwargs)
    fname_dst = get_ionex_filename(date_str, tec_dir=tec_dir, **kwargs) + ".Z"
    fname_dst_uncomp = fname_dst[:-2]

    # download - compose cmd
    cmd = f'wget --continue --auth-no-challenge "{fname_src}"'
    if os.path.isfile(fname_dst) and os.path.getsize(fname_dst) > 1000:
        cmd += " --timestamping"

    # Record executed command line in logging file
    logging.info(f"Execute command: {cmd}")

    # download - run cmd in output dir
    pwd = os.getcwd()
    os.chdir(tec_dir)
    os.system(cmd)
    os.chdir(pwd)

    # uncompress
    # if output file 1) does not exist or 2) smaller than 400k in size or 3) older
    if (
        not os.path.isfile(fname_dst_uncomp)
        or os.path.getsize(fname_dst_uncomp) < 400e3
        or os.path.getmtime(fname_dst_uncomp) < os.path.getmtime(fname_dst)
    ):
        cmd = f"gzip --force --keep --decompress {fname_dst}"
        logging.info(f"Execute command: {cmd}")
        os.system(cmd)

    return fname_dst_uncomp


def get_ionex_filename(date_str, tec_dir=None, sol_code="jpl", date_fmt="%Y%m%d"):
    """
    Get the file name of the IONEX file

    Parameters
    ----------
    date_str: str
        Date in the 'date_fmt' format
    tec_dir: str
        Directory where to store downloaded TEC files
    sol_code: str
        GIM analysis center code in 3 digits
        (values: cod, esa, igs, jpl, upc, uqr)
    date_fmt: str
        Date format string

    Returns
    -------
    tec_file: str
        Path to the local uncompressed (or remote compressed) TEC file
    """

    dd = dt.datetime.strptime(date_str, date_fmt)
    doy = f"{dd.timetuple().tm_yday:03d}"
    yy = str(dd.year)[2:4]

    # file name base
    fname = f"{sol_code.lower()}g{doy}0.{yy}i.Z"

    # full path
    if tec_dir:
        # local uncompressed file path
        tec_file = os.path.join(tec_dir, fname[:-2])
    else:
        # remote compressed file path
        url_dir = "https://cddis.nasa.gov/archive/gnss/products/ionex"
        tec_file = os.path.join(url_dir, str(dd.year), doy, fname)

    return tec_file


def get_ionex_height(tec_file):
    """
    Get the height of the thin-shell ionosphere from IONEX file

    Parameters
    ----------
    tec_file: str
        Path to the TEC file in IONEX format

    Returns
    -------
    iono_hgt: float
        Height above the surface in meters
    """

    with open(tec_file) as f:
        lines = f.readlines()
        for line in lines:
            if line.strip().endswith("DHGT"):
                ion_hgt = float(line.split()[0])
                break
    return ion_hgt


def ionosphere_delay(utc_time, wavelength, tec_file, lon_arr, lat_arr, inc_arr):
    """
    Calculate ionosphere delay for geolocation

    Parameters
    ----------
    utc_time: datetime.datetime
        UTC time to calculate the ionosphere delay
    wavelength: float
        Wavelength of the signal
    tec_file: str
        Path to the TEC file
    lon_arr: numpy.ndarray
        array of longitude in radar grid. unit: degrees
    lat_arr: numpy.ndarray
        array of latitude in radar grid. unit: degrees
    inc_arr: numpy.ndarray
        array of incidence angle in radar grid. unit: degrees

    Returns
    -------
    los_iono_delay: np.ndarray
        Ionospheric delay in line of sight in meters
    """

    if not tec_file:
        print('"tec_file" was not provided. Ionosphere correction will not be applied.')
        return np.zeros(lon_arr.shape)

    if not os.path.exists(tec_file):
        raise RuntimeError(f"IONEX file was not found: {tec_file}")

    utc_tod_sec = utc_time.hour * 3600.0 + utc_time.minute * 60.0 + utc_time.second

    ionex_val = get_ionex_value(
        tec_file, utc_tod_sec, lat_arr.flatten(), lon_arr.flatten()
    )

    ionex_val = ionex_val.reshape(lon_arr.shape)

    freq_sensor = isce3.core.speed_of_light / wavelength

    # Constant to convert the TEC product to electrons / m2
    ELECTRON_PER_SQM = ionex_val * 1e16

    # Constant in m3/s2
    K = 40.31

    los_iono_delay = K * ELECTRON_PER_SQM / freq_sensor**2 / np.cos(np.deg2rad(inc_arr))

    return los_iono_delay

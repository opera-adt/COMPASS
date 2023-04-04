import argparse
import os

import h5py
import isce3
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysolid
import shapely.wkt as wkt
from osgeo import gdal
from pyproj import CRS, Proj
from shapely import geometry

dateformat = '%Y-%m-%d %H:%M:%S.%f'

def run(cslc_file, cr_file, csv_output_file=None, plot_ale=False,
        correct_set=True):
    '''
    Compute absolute geolocation error based on CSLC-S1
    product and a file containing surveying corner
    reflector (CR) locations

    Parameters
    ----------
    cslc_file: str
        File path to HDF5 CSLC-S1 product
    cr_file: str
        File path to CSV file containing
        surveyed CR locations
    csv_output_file: str, None
        File path to save computations in
        CSV file. If None, csv file is not saved
    plot_ale: bool
        If True, plots Absolute geolocation error
        results
    correct_set: bool
        Correct Corner Reflector positions for Solid
        Earth tides
    '''

    # Check that the CSLC-S1 product file exists
    if not os.path.exists(cslc_file):
        err_str = f'{cslc_file} CSLC-S1 product does not exist'
        raise FileNotFoundError(err_str)

    # Check corner reflector file exists
    if not os.path.exists(cr_file):
        err_str = f'{cr_file} corner reflector file does not exist'
        raise FileNotFoundError(err_str)

    # Open corner reflector file
    cr_df = pd.read_csv(cr_file)

    # Add empty columns necessary for computation

    # Identify which CR are contained in the usable part of the burst
    # Get CSLC-S1 polygon
    cslc_poly = get_cslc_polygon(cslc_file)
    cr_x = []
    cr_y = []
    cr_x_cslc = []
    cr_y_cslc = []

    for idx, row in cr_df.iterrows():
        cr_lat = row['Latitude (deg)']
        cr_lon = row['Longitude (deg)']
        cr_loc = geometry.Point(cr_lon, cr_lat)

        if cslc_poly.contains(cr_loc):
            # Convert corner lat/lon coordinates in UTM
            cslc_epsg = get_cslc_epsg(cslc_file)

            # Correct corner reflector position for solid Earth tides
            # otherwise just transform coordinates to UTM
            if correct_set:
                x, y = correct_cr_tides(cslc_file, cr_lat, cr_lon)
            else:
                x, y = latlon2utm(cr_lat, cr_lon, cslc_epsg)

            cr_x.append(x)
            cr_y.append(y)

            # Compute location of CR in CSLC image at pixel precision
            x_coord_vect, dx, y_coord_vect, dy = get_xy_info(cslc_file)
            cr_x_cslc.append(int((x - x_coord_vect[0]) / dx))
            cr_y_cslc.append(int((y - y_coord_vect[0]) / dy))
        else:
            # If the CR is not in the burst, drop the corresponding
            # row from the panda dataframe
            cr_df.drop(idx, inplace=True)

    # Assign computed data
    cr_df['CR_X'] = cr_x
    cr_df['CR_Y'] = cr_y
    cr_df['CR_X_CSLC'] = cr_x_cslc
    cr_df['CR_Y_CSLC'] = cr_y_cslc

    x_peak_vect = []
    y_peak_vect = []
    
    # Find peak location for every corner reflector in DataFrame
    cslc_arr = get_cslc(cslc_file)
    for idx, row in cr_df.iterrows():
        x_peak, y_peak = find_peak(cslc_arr, int(row['CR_X_CSLC']),
                                   int(row['CR_Y_CSLC']))
        x_coord_vect, dx, y_coord_vect, dy = get_xy_info(cslc_file)
        x_peak_vect.append(x_coord_vect[0] + x_peak * dx)
        y_peak_vect.append(y_coord_vect[0] + y_peak * dy)

    cr_df['CR_X_CSLC_PEAK'] = x_peak_vect
    cr_df['CR_Y_CSLC_PEAK'] = y_peak_vect

    # Compute absolute geolocation error along X and Y direction
    cr_df['ALE_X'] = cr_df['CR_X_CSLC_PEAK'] - cr_df['CR_X']
    cr_df['ALE_Y'] = cr_df['CR_Y_CSLC_PEAK'] - cr_df['CR_Y']

    if csv_output_file is not None:
        cr_df.to_csv(csv_output_file)

    if plot_ale:
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(cr_df['ALE_Y'], cr_df['ALE_X'], s=200, alpha=0.8,
                        marker='o')

        ax.grid(True)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.axhline(0, color='black')
        ax.axvline(0, color='black')
        ax.set_xlabel('Easting error (m)')
        ax.set_ylabel('Northing error (m)')
        fig.suptitle('Absolute geolocation error')
        plt.show()


def correct_cr_tides(cslc_file, cr_lat, cr_lon):
    '''
    Correct Corner reflector position for Solid Earth tides
    Parameters
    ----------
    cslc_file: str
        File path to CSLC product
    x_cr: float
        Corner reflector position along X-direction
    y_cr: float
        Corner reflector position along Y-direction

    Returns
    -------
    x_tide_cr: float
        Corner reflector position along X-direction corrected
        for Solid Earth tides
    y_tide_cr: float
        Corner reflector position along Y-direction corrected
        for Solid Earth tide
    '''

    # Get burst sensing_start and sensing stop
    with h5py.File(cslc_file, 'r') as h5:
        start = h5['/science/SENTINEL1/CSLC/metadata/processing_information/s1_burst_metadata/sensing_start'][()]
        stop = h5['/science/SENTINEL1/CSLC/metadata/processing_information/s1_burst_metadata/sensing_stop'][()]

    sensing_start = dt.datetime.strptime(str(start), dateformat)
    sensing_stop = dt.datetime.strptime(str(stop), dateformat)

    # Call pysolid to compute the Solid Earth tides
    (dt_out,
     tide_e,
     tide_n,
     tide_u) = pysolid.calc_solid_earth_tides_point(cr_lat, cr_lon, sensing_start,
                                                    sensing_stop,
                                                    step_sec=5,
                                                    display=False,
                                                    verbose=False)
    tide_e = np.mean(tide_e[0:2])
    tide_n = np.mean(tide_n[0:2])

    # Transform CR coordinates to UTM
    cslc_epsg = get_cslc_epsg(cslc_file)
    x, y = latlon2utm(cr_lat, cr_lon, cslc_epsg)

    return x + tide_e , y + tide_n


def find_peak(arr, x_loc, y_loc, ovs_factor=64, margin=8):
    '''
    Find peak location in 'arr'
    Parameters
    ----------
    arr: np.ndarray
        Array to use for identifying peak location
    x_loc: np.int
        First guess of peak location along X-coordinates
    y_loc: np.int
        First guess of peak location along Y-coordinates
    ovs_factor: np.int
        Oversampling factor
    margin: int
        Margin

    Returns
    -------
    x_peak: np.float
        Peak location along X-coordinates
    y_peak: np.float
        Peak location along Y-coordinate
    '''

    # Extract an area around x_loc, y_loc
    img = arr[(y_loc - margin):(y_loc + margin),
              (x_loc - margin):(x_loc + margin)]

    # Oversample CSLC subset and get amplitude
    img_ovs = isce3.cal.point_target_info.oversample(
        img, ovs_factor)
    idx_peak_ovs = np.argmax(np.abs(img_ovs))
    img_peak_ovs = np.unravel_index(idx_peak_ovs, img_ovs.shape)

    # Get location of the peak wrt the upper left corner of image subset
    x_peak = x_loc - margin + img_peak_ovs[1] / ovs_factor
    y_peak = y_loc - margin + img_peak_ovs[0] / ovs_factor

    return x_peak, y_peak


def get_cslc(cslc_file, pol='VV'):
    '''
    Get CSLC-S1 array associated to 'pol'

    Parameters
    ----------
    cslc_file: str
        File path to CSLC-S1 product
    pol: str
        Polarization string of CSLC-S1 array
        to extract from product

    Returns
    -------
    cslc: np.ndarray
        CSLC-S1 image associated to polarization
        channel 'pol'
    '''
    with h5py.File(cslc_file, 'r') as h5:
        cslc = h5[f'science/SENTINEL1/CSLC/grids/{pol}'][()]
    return cslc


def get_xy_info(cslc_file, pol='VV'):
    '''
    Get X/Y-coordinate vector and spacings
    from CSLC-S1 product in 'cslc_file'

    Parameters
    ----------
    cslc_file: str
        File path to CSLC-S1 product

    Returns
    -------
    x_vect: np.ndarray
        Array of X-coordinates associated to CSLC-S1
    y_vect: np.ndarray
        Array of Y-coordinates associated to CSLC-S1
    x_spac: np.float
        CSLC-S1 spacing along X-direction
    y_spac: np.float
        CSLC-S1 spacing along Y-direction
    '''
    # Open CSLC with NETCDF4 driver
    ds_in = gdal.Open(f'NETCDF:{cslc_file}:science/SENTINEL1/CSLC/grids/{pol}')
    width = ds_in.RasterXSize
    length = ds_in.RasterYSize

    geo_trans = ds_in.GetGeoTransform()
    x_spac = geo_trans[1]
    y_spac = geo_trans[5]

    # Generate x_vect and y_vect
    x_vect = geo_trans[0] + np.arange(0, width)*x_spac
    y_vect = geo_trans[1] + np.arange(0, length)*y_spac

    return x_vect, x_spac, y_vect, y_spac


def latlon2utm(lat, lon, out_epsg):
    '''
    Converts lat/lon to x/y coordinates
    specified by 'out_epsg'

    Parameters
    ----------
    lat: np.float
        Latitude coordinates
    lon: np.float
        Longitude coordinates
    out_epsg: int
        EPSG code identifying coordinate system
        for lat/lon conversion

    Returns
    -------
    x: np.float
        X-coordinate
    y: np.float
        Y-coordinate
    '''

    _proj = Proj(CRS.from_epsg(out_epsg))
    x, y = _proj(lon, lat, inverse=False)
    return x, y


def get_cslc_polygon(cslc_file):
    '''
    Get the polygon containing the valid values of
    the CSLC-S1 product in 'cslc_file'

    Parameters
    ----------
    cslc_file: str
        File path to the CSLC-S1 product

    Returns
    -------
    cslc_poly: shapely.Polygon
        Shapely polygon including CSLC-S1 valid values
    '''
    with h5py.File(cslc_file, 'r') as h5:
        poly = h5['science/SENTINEL1/identification/bounding_polygon'][()]
    cslc_poly = wkt.loads(poly)
    return cslc_poly


def get_cslc_epsg(cslc_file):
    '''
    Returns projection code for CSLC-S1 product
    in 'cslc_file'

    Parameters
    ----------
    cslc_file: str
        Path to CSLC-S1 product

    Returns
    -------
    epsg: int
        EPSG code identifying the projection of CSLC-S1
    '''
    with h5py.File(cslc_file, 'r') as h5:
        epsg = h5['/science/SENTINEL1/CSLC/grids/projection'][()]
    return epsg


def create_parser():
    '''
    Generate command line parser
    '''

    parser = argparse.ArgumentParser(
        description="Compute absolute geolocation error (ALE) for CSLC-S1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument('-p', '--cslc-s1', required=True, dest='cslc_file',
                          help='OPERA L2 CSLC-S1 product')
    required.add_argument('-c', '--cr-file', required=True, dest='cr_file',
                          help='Corner reflector position CSV file '
                               'same date as input CSLC-S1')
    optional.add_argument('-s', '--save-csv', dest='save_csv', default=None,
                          help='Save ALE in CSV file (default: None)')
    optional.add_argument('-i', '--plot-ale', dest='plot_ale', default=False,
                          help='Plot ALE')
    optional.add_argument('-t', '--set', dest=set, default=False,
                          help='If True, correct corner reflector positions for Solid Earth Tides')
    return parser.parse_args()


def main():
    '''
    Create command line interface and run geolocation error script
    '''

    args = create_parser()
    run(cslc_file=args.cslc_file,
        cr_file=args.cr_file,
        csv_output_file=args.save_csv,
        plot_ale=args.plot_ale,
        correct_set=args.set)


if __name__ == '__main__':
    main()

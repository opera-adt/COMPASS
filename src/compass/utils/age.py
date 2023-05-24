import argparse
import os

import h5py
import isce3
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.wkt as wkt
from osgeo import gdal, osr
from pyproj import CRS, Proj
from shapely import geometry
from compass.utils.h5_helpers import (DATA_PATH,
                                      METADATA_PATH,
                                      TIME_STR_FMT)


def run(cslc_file, cr_file, csv_output_file=None, plot_age=False,
        correct_set=False, mission_id='S1', pol='VV', ovs_factor=128,
        margin=32, apply_az_ramp=True, unflatten=True):
    '''
    Compute Absolute Geolocation Error (AGE) for geocoded SLC
    products from Sentinel-1 or NISAR missions. AGE is computed
    by differencing the surveyed corner reflector (CR) positions from
    the CSV file with that detected in the geocoded CSLC product.

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
    plot_age: bool
        If True, plots Absolute geolocation error
        results
    correct_set: bool
        Correct Corner Reflector positions for Solid
        Earth tides
    mission_id: str
        Mission identifier. 'S1' for Sentinel-1 OPERA
        CSLC or 'NI' for NISAR GSLC
    pol: str
        Polarization channel to use to evaluate AGE
    ovs_factor: int
        Oversampling factor detecting CR location with
        sub-pixel accuracy (default: 128)
    margin: int
        Margin to consider around CR position detected in
        the geocoded SLC image. Overall included margin
        is 2*margin from left-to-right and from top-to-bottom
        (default: 32)
    apply_az_ramp: bool
        If set to True, removes the azimuth carrier ramp prior
        to detect CR peak. Set this option to True for
        S1-A/B data where deramping has an effect on AGE.
        Azimuth carrier ramp is extracted from CSLC-S1 products
    unflatten: bool
        If set to True, adds back the flatten phase prior to detect
        CR peak. Set this option to True for AGE computation as it
        drammatically affect the correct determination of the peak
        location
    '''

    # Check that the CSLC-S1 product file exists
    if not os.path.exists(cslc_file):
        err_str = f'{cslc_file} input geocoded SLC product does not exist'
        raise FileNotFoundError(err_str)

    # Check corner reflector file exists
    if not os.path.exists(cr_file):
        err_str = f'{cr_file} CSV CR position file does not exist'
        raise FileNotFoundError(err_str)

    # Open and load CSV CR file in pandas dataframe
    cr_df = pd.read_csv(cr_file)

    # Identify CRs contained in the usable part of the
    # geocoded SLC
    cslc_poly = get_cslc_polygon(cslc_file,
                                 mission_id=mission_id)
    # Initialize empty lists to include in the CR
    # pandas dataframe
    cr_x = []
    cr_y = []
    cr_x_cslc = []
    cr_y_cslc = []

    for idx, row in cr_df.iterrows():
        # Extract surveyed CR positions from pandas dataframe
        cr_lat = row['Latitude (deg)']
        cr_lon = row['Longitude (deg)']
        cr_loc = geometry.Point(cr_lon, cr_lat)

        # Add buffer of approx. 30 m to CR location
        buff_cr_loc = cr_loc.buffer(0.0003)

        # If the CR is contained in the geocoded SLC product
        # get its position in the SLC image, otherwise, drop
        # the CR from the pandas dataframe
        if cslc_poly.contains(buff_cr_loc):
            # Convert corner lat/lon coordinates in UTM
            cslc_epsg = get_cslc_epsg(cslc_file, mission_id=mission_id,
                                      pol=pol)

            # Correct corner reflector position for solid Earth tides
            # otherwise just transform coordinates to UTM
            if correct_set:
                x, y = correct_cr_tides(cslc_file, cr_lat, cr_lon,
                                        mission_id=mission_id, pol=pol)
            else:
                x, y = latlon2utm(cr_lat, cr_lon, cslc_epsg)

            cr_x.append(x)
            cr_y.append(y)

            # Compute CR location in the geocoded SLC at pixel-precision
            x_start, dx, y_start, dy = get_xy_info(cslc_file,
                                                   mission_id=mission_id,
                                                   pol=pol)
            cr_x_cslc.append(int((x - x_start) / dx))
            cr_y_cslc.append(int((y - y_start) / dy))
        else:
            # The CR is not contained in the geocoded CSLC; drop it
            # from the pandas dataframe
            cr_df.drop(idx, inplace=True)

    # Assign computed data
    cr_df['CR_X'] = cr_x
    cr_df['CR_Y'] = cr_y
    cr_df['CR_X_CSLC'] = cr_x_cslc
    cr_df['CR_Y_CSLC'] = cr_y_cslc

    x_peak_vect = []
    y_peak_vect = []
    cr_snr_vect = []

    # Find peak location for every corner reflector in DataFrame
    # Open CSLC and apply deramping and flattening if desired
    arr = get_cslc(cslc_file,
                   mission_id=mission_id,
                   pol=pol)
    # If True, remove azimuth carrier ramp
    if apply_az_ramp:
        carrier_phase = get_carrier_phase(cslc_file, mission_id=mission_id)
        arr *= np.exp(-1j * carrier_phase)

    # If True, adds back flattening phase
    if unflatten:
        flatten_phase = get_flatten_phase(cslc_file, mission_id=mission_id)
        arr *= np.exp(-1j * flatten_phase)

    for idx, row in cr_df.iterrows():
        x_peak, y_peak, snr_cr = find_peak(arr, cslc_file, int(row['CR_X_CSLC']),
                                           int(row['CR_Y_CSLC']), pol=pol,
                                           mission_id=mission_id, ovs_factor=ovs_factor,
                                           margin=margin)

        x_peak_vect.append(x_peak)
        y_peak_vect.append(y_peak)
        cr_snr_vect.append(snr_cr)

    cr_df['CR_X_CSLC_PEAK'] = x_peak_vect
    cr_df['CR_Y_CSLC_PEAK'] = y_peak_vect
    cr_df['CR_SNR'] = cr_snr_vect

    # Compute absolute geolocation error along X and Y direction
    cr_df['ALE_X'] = cr_df['CR_X_CSLC_PEAK'] - cr_df['CR_X']
    cr_df['ALE_Y'] = cr_df['CR_Y_CSLC_PEAK'] - cr_df['CR_Y']

    if csv_output_file is not None:
        cr_df.to_csv(csv_output_file)
    else:
        print('Print to screen AGE results')
        print(cr_df)

    if plot_age:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(cr_df['ALE_X'], cr_df['ALE_Y'], s=200, alpha=0.8,
                   marker='o')

        ax.grid(True)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.axhline(0, color='black')
        ax.axvline(0, color='black')
        ax.set_xlabel('Easting error (m)')
        ax.set_ylabel('Northing error (m)')
        fig.suptitle('Absolute geolocation error (AGE)')
        plt.show()


def correct_cr_tides(cslc_file, cr_lat, cr_lon,
                     mission_id='S1', pol='VV'):
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
    mission_id: str
        Mission identifier. S1: Sentinel or NI: NISAR

    Returns
    -------
    x_tide_cr: float
        Corner reflector position along X-direction corrected
        for Solid Earth tides
    y_tide_cr: float
        Corner reflector position along Y-direction corrected
        for Solid Earth tide
    '''
    import pysolid
    # Get geocode SLC sensing start and stop
    if mission_id == 'S1':
        start_path = f'{METADATA_PATH}/processing_information/s1_burst_metadata/sensing_start'
        stop_path = f'{METADATA_PATH}/processing_information/s1_burst_metadata/sensing_stop'
    elif mission_id == 'NI':
        start_path = '/science/LSAR/GSLC/identification/zeroDopplerStartTime'
        stop_path = '/science/LSAR/GSLC/identification/zeroDopplerEndTime'
    else:
        err_str = f'{mission_id} is not a valid mission identifier'
        raise ValueError(err_str)

    with h5py.File(cslc_file, 'r') as h5:
        start = h5[start_path][()]
        stop = h5[stop_path][()]

    sensing_start = dt.datetime.strptime(start.decode('UTF-8'),
                                         TIME_STR_FMT)
    sensing_stop = dt.datetime.strptime(stop.decode('UTF-8'),
                                        TIME_STR_FMT)

    # Compute SET in ENU using pySolid
    (_,
     tide_e,
     tide_n,
     _) = pysolid.calc_solid_earth_tides_point(cr_lat, cr_lon,
                                                    sensing_start,
                                                    sensing_stop,
                                                    step_sec=5,
                                                    display=False,
                                                    verbose=False)
    tide_e = np.mean(tide_e[0:2])
    tide_n = np.mean(tide_n[0:2])

    # Transform CR coordinates to UTM
    cslc_epsg = get_cslc_epsg(cslc_file,
                              mission_id=mission_id,
                              pol=pol)
    x, y = latlon2utm(cr_lat, cr_lon, cslc_epsg)
    x_tide_cr = x + tide_e
    y_tide_cr = y + tide_n

    return x_tide_cr, y_tide_cr


def find_peak(arr, cslc_file, x_loc, y_loc, pol='VV',
              mission_id='S1', ovs_factor=128, margin=32):
    '''
    Find peak location in 'arr'
    Parameters
    ----------
    arr: np.ndarray
        Array to use for peak determination
    cslc_file: str
        File path to CSLC product
    x_loc: np.int
        First guess of peak location along X-coordinates
    y_loc: np.int
        First guess of peak location along Y-coordinates
    mission_id: str
        Mission identifier. S1: Sentinel-1 or NI: NISAR
    pol: str
        Polarization to use for peak detection
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
    snr_cr_db: np.float
        Peak SNR for identified corner reflector
    '''

    x_start, x_spac, y_start, y_spac = get_xy_info(cslc_file,
                                                   mission_id=mission_id,
                                                   pol=pol)

    # Check if the X/Y coordinate in the image are withing the input CSLC
    upperleft_x = int(np.round(x_loc)) - margin // 2
    upperleft_y = int(np.round(y_loc)) - margin // 2
    lowerright_x = upperleft_x + margin
    lowerright_y = upperleft_y + margin

    if (upperleft_x < 0) or (upperleft_y < 0) or \
            (lowerright_x > arr.shape[1]) or (lowerright_y > arr.shape[0]):
        err_msg = 'The corner reflector input coordinates are outside of the CSLC'
        raise ValueError(err_msg)

    # Extract an area around x_loc, y_loc
    img = arr[upperleft_y:lowerright_y, upperleft_x:lowerright_x]

    # Check if the SNR of the peak is above the threshold
    snr_cr_db = get_snr_cr(img)

    # Oversample CSLC subset and get amplitude
    img_ovs = isce3.cal.point_target_info.oversample(
        img, ovs_factor)
    idx_peak_ovs = np.argmax(np.abs(img_ovs))
    img_peak_ovs = np.unravel_index(idx_peak_ovs, img_ovs.shape)

    x_chip = x_start + x_spac * upperleft_x
    y_chip = y_start + y_spac * upperleft_y

    dx1 = x_spac / ovs_factor
    dy1 = y_spac / ovs_factor

    x_cr = x_chip + x_spac / 2 + img_peak_ovs[1] * dx1
    y_cr = y_chip + y_spac / 2 + img_peak_ovs[0] * dy1

    return x_cr, y_cr, snr_cr_db


def get_carrier_phase(cslc_file, mission_id='S1'):
    '''
    Get azimuth carrier phase from CSLC product
    Note, this is implemented only for CSLC-S1

    Parameters
    ----------
    cslc_file: str
        File path to CSLC product
    mission_id: str
        Mission identifier. S1: Sentinel-1 or
        NI: NISAR (default: S1)

    Returns
    -------
    carrier_phase: np.ndarray, float64
        Numpy array containing azimuth carrier phase
    '''

    if mission_id == 'S1':
        carrier_path = f'{DATA_PATH}/azimuth_carrier_phase'
    else:
        err_str = f"Azimuth carrier phase not present for {mission_id} CSLC product"
        raise ValueError(err_str)

    with h5py.File(cslc_file, 'r') as h5:
        carrier_phase = h5[carrier_path][()]
    return carrier_phase


def get_flatten_phase(cslc_file, mission_id='S1'):
    '''
    Get flattening phase from CSLC product
    Note, this is implemented only for CSLC-S1

    Parameters
    ----------
    cslc_file: str
        File path to CSLC product
    mission_id: str
        Mission identifier. S1: Sentinel-1 or
        NI: NISAR (default: S1)

    Returns
    -------
    flatten_phase: np.ndarray, float64
        Numpy array containing flattening phase
    '''

    if mission_id == 'S1':
        rg_off_path = f'{DATA_PATH}/flattening_phase'
    else:
        err_str = f"Range offsets not present for {mission_id} CSLC product"
        raise ValueError(err_str)

    with h5py.File(cslc_file, 'r') as h5:
        flatten_phase = h5[rg_off_path][()]

    return flatten_phase


def get_cslc(cslc_file, mission_id='S1', pol='VV') -> np.ndarray :
    '''
    Get CSLC-S1 array associated to 'pol'

    Parameters
    ----------
    cslc_file: str
        File path to CSLC product
    mission_id: str
        Mission identifier. S1: Sentinel-1 or
        NI: NISAR (default: S1)
    pol: str
        Polarization string of CSLC-S1 array
        to extract from product

    Returns
    -------
    cslc: np.ndarray
        Geocoded SLC image corresponding to 'pol'
        polarization channel.
    '''

    if mission_id == 'S1':
        cslc_path = f'{DATA_PATH}/{pol}'
    elif mission_id == 'NI':
        with h5py.File(cslc_file, 'r') as h5:
             frequencies = h5["/science/LSAR/identification/listOfFrequencies"][()]
             freq = frequencies[0].decode('utf-8')
             frequency = f'frequency{freq}'
        cslc_path = f'/science/LSAR/GSLC/grids/{frequency}/{pol}'
    else:
        err_str = f'{mission_id} is not a valid mission identifier'
        raise ValueError(err_str)

    with h5py.File(cslc_file, 'r') as h5:
        cslc = h5[cslc_path][()]

    return cslc


def get_xy_info(cslc_file, mission_id='S1', pol='VV'):
    '''
    Get X/Y spacings and coordinate vectors from the
    geocoded SLC contained in 'cslc_file'

    Parameters
    ----------
    cslc_file: str
        File path to CSLC-S1 product
    mission_id: str
        Mission identifier. S1: Sentinel-1 or
        NI: NISAR (default: S1)
    pol: str
        Polarization channel of the geocoded SLC
        to analyze (default: VV)

    Returns
    -------
    x_start: np.float
        X-coordinate of the upper-left corner of the upper-left pixel
    y_start: np.float
        Y-coordinate of the upper-left corner of the upper-left pixel
    x_spac: np.float
        CSLC-S1 spacing along X-direction
    y_spac: np.float
        CSLC-S1 spacing along Y-direction
    '''
    if mission_id == 'S1':
        cslc_path = DATA_PATH
    elif mission_id == 'NI':
        cslc_path = '/science/LSAR/GSLC/grids/frequencyA'
    else:
        err_str = f'{mission_id} is not a valid mission identifier'
        raise ValueError(err_str)

    # Open geocoded SLC with a NetCDF driver
    ds_in = gdal.Open(f'NETCDF:{cslc_file}:{cslc_path}/{pol}')

    geo_trans = ds_in.GetGeoTransform()
    x_spac = geo_trans[1]
    y_spac = geo_trans[5]

    # Generate x_vect and y_vect
    x_start = geo_trans[0]
    y_start = geo_trans[3]

    return x_start, x_spac, y_start, y_spac


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


def get_cslc_polygon(cslc_file, mission_id='S1'):
    '''
    Get bounding polygon identifying the valid portion
    of the geocoded SLC product on the ground

    Parameters
    ----------
    cslc_file: str
        File path to the CSLC-S1 product
    mission_id: str
        Mission identifier; S1: Sentinel-1
        NI: NISAR

    Returns
    -------
    cslc_poly: shapely.Polygon
        Shapely polygon including CSLC-S1 valid values
    '''
    if mission_id == 'S1':
        poly_path = 'identification/bounding_polygon'
    elif mission_id == 'NI':
        poly_path = 'science/LSAR/identification/boundingPolygon'
    else:
        err_str = f'{mission_id} is not a valid mission identifier'
        raise ValueError(err_str)

    with h5py.File(cslc_file, 'r') as h5:
        poly = h5[poly_path][()]

    cslc_poly = wkt.loads(poly.decode('UTF-8'))

    return cslc_poly


def get_cslc_epsg(cslc_file, mission_id='S1', pol='VV'):
    '''
    Returns EPSG projection code for geocoded
    SLC contained in 'cslc_file'

    Parameters
    ----------
    cslc_file: str
        Path to CSLC-S1 product
    mission_id: str
        Mission identifier. S1: Sentinel-1
        NI: NISAR
    pol: str
        Polarization channel

    Returns
    -------
    epsg: int
        EPSG code identifying the projection of the
        geocoded SLC product
    '''
    if mission_id == 'S1':
        epsg_path = f'{DATA_PATH}/projection'
        with h5py.File(cslc_file, 'r') as h5:
            epsg = h5[epsg_path][()]
    elif mission_id == 'NI':
        with h5py.File(cslc_file, 'r') as h5:
            frequencies = h5["/science/LSAR/identification/listOfFrequencies"]
            freq = frequencies[0].decode('utf-8')
            frequency = f'frequency{freq}'
        dataset_path = f'NETCDF:{cslc_file}://science/LSAR/GSLC/grids/{frequency}/{pol}'
        ds = gdal.Open(dataset_path, gdal.GA_ReadOnly)
        s = osr.SpatialReference(wkt=ds.GetProjection()).ExportToProj4()
        crs = CRS.from_proj4(s)
        epsg = crs.to_epsg()
    else:
        err_str = f'{mission_id} is not a valid mission identifier'
        raise ValueError(err_str)

    return epsg


def get_snr_cr(img: np.ndarray, cutoff_percentile: float=3.0):
    '''
    Estimate the signal-to-noise ration (SNR) of the corner reflector contained in img
    in the input image patch

    Parameter
    ---------
    img: numpy.ndarray
        SLC image patch to calculate the SNR
    cutoff_percentile: float
        Cutout ratio of high and low part of the signal to cutoff

    Returns
    -------
    snr_cr_db: float
        SNR of the peak in decibel (db)
    '''

    power_arr = img.real ** 2 + img.imag ** 2

    # build up the mask array
    thres_low = np.nanpercentile(power_arr, cutoff_percentile)
    thres_high = np.nanpercentile(power_arr, 100 - cutoff_percentile)
    mask_threshold = np.logical_and(power_arr < thres_low,
                                    power_arr > thres_high)
    mask_invalid_pixel = np.logical_and(power_arr <= 0.0,
                                        np.isnan(power_arr))
    ma_power_arr = np.ma.masked_array(power_arr,
                                      mask=np.logical_and(mask_threshold,
                                                          mask_invalid_pixel))

    peak_power = power_arr.max()
    mean_background_power = np.mean(ma_power_arr)

    snr_cr_db = np.log10(peak_power / mean_background_power) * 10.0

    return snr_cr_db


def create_parser():
    '''
    Generate command line parser
    '''

    parser = argparse.ArgumentParser(
        description="Compute absolute geolocation error (AGE) for geocoded SLC"
                    "from Sentinel-1 or NISAR missions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument('-p', '--cslc-s1', required=True, dest='cslc_file',
                          help='File path to geocoded SLC product')
    required.add_argument('-c', '--cr-file', required=True, dest='cr_file',
                          help='File path to CSV corner reflector position file')
    optional.add_argument('-s', '--save-csv', dest='save_csv', default=None,
                          help='File path to save AGE results in CSV format')
    optional.add_argument('-i', '--plot-age', dest='plot_age', default=False,
                          help='If True, plots AGE results ')
    optional.add_argument('-t', '--set', dest='set', default=False,
                          help='If True, corrects CSV corner reflector positions for Solid Earth Tides (default: False)')
    optional.add_argument('-m', '--mission-id', dest='mission_id', default='S1',
                          help='Mission identifier; S1: Sentinel1, NI: NISAR')
    optional.add_argument('-pol', '--polarization', dest='pol', default='VV',
                          help='Polarization channel to use to evaluate AGE ')
    optional.add_argument('-o', '--ovs', dest='ovs_factor', default=128, type=int,
                          help='Oversample factor for determining CR location in the '
                               'geocoded SLC with sub-pixel accuracy')
    optional.add_argument('-mm', '--margin', dest='margin', default=32, type=int,
                          help='Padding margin around CR position detected in the geocoded SLC '
                               'image. Actual margin is 2*margin from left-to-right and from'
                               'top-to-bottom')
    optional.add_argument('-r', '--azimuth-ramp', dest='apply_az_ramp', default=True,
                          help='If True, removes azimuth carrier ramp prior to CR peak location. '
                               'This option should be set to True for S1-A/B AGE analyses')
    optional.add_argument('-u', '--unflatten', dest='unflatten', default=True,
                          help='If True, adds back the flatten phase. This option should be set'
                               'to True if the geocoded SLC product has not been flattened')
    return parser.parse_args()


def main():
    '''
    Create command line interface and run geolocation error script
    '''

    args = create_parser()
    run(cslc_file=args.cslc_file,
        cr_file=args.cr_file,
        csv_output_file=args.save_csv,
        plot_age=args.plot_age,
        correct_set=args.set,
        mission_id=args.mission_id,
        pol=args.pol,
        ovs_factor=args.ovs_factor,
        margin=args.margin,
        apply_az_ramp=args.apply_az_ramp,
        unflatten=args.unflatten)


if __name__ == '__main__':
    main()

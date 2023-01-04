'''
Test IONEX functionalities: file reading and interpolation
'''

import os
import numpy as np

from compass.utils import iono

# Set the path to fetch data for the test
tec_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests/data")
date_str = '20151115'
sol_code = 'jpl'


def prep_test_data(download_data=True):
    '''
    Prepare IONEX data for unit test

    Parameters
    ----------
    download_data: bool
        Boolean flag allow to download TEC data
        for unit test if set to True

    Returns
    -------
    tec_file: str
        Path to local or downloaded TEC file to
        use in the unit test
    '''

    # Create TEC directory
    os.makedirs(tec_dir, exist_ok=True)

    # Generate the TEC filename
    tec_file = iono.get_ionex_filename(
        date_str,
        tec_dir=tec_dir,
        sol_code=sol_code
    )

    # If prep_mode=True, download data
    if download_data:
        if not os.path.isfile(tec_file):
            print(f'Download IONEX file at {date_str} from {sol_code} to {tec_dir}')
            tec_file = iono.download_ionex(date_str,
                                           tec_dir,
                                           sol_code=sol_code)
    return tec_file


def test_read_ionex():
    '''
    Test the reader for IONEX data
    '''

    time_ind = 1
    x0, x1, y0, y1 = 3, 9, 28, 33

    # Create TEC data on a region of interest (AOI)
    tec_aoi = np.array(
        [[71.8, 64.2, 55.9, 47.1, 38.6, 31.2],
         [80.2, 73.9, 66.6, 58.2, 49.5, 41.1],
         [83.2, 79.6, 74.6, 68., 60.1, 51.6],
         [79.6, 79.5, 78.1, 74.5, 68.5, 60.9],
         [71.9, 74.5, 76.5, 76.2, 73.1, 67.3]],
    )

    # Get IONEX file path
    tec_file = iono.get_ionex_filename(
        date_str,
        tec_dir=tec_dir,
        sol_code=sol_code,
    )

    # Read IONEX data
    mins, lats, lons, tec_maps = iono.read_ionex(tec_file)[:4]
    assert np.allclose(tec_maps[time_ind, y0:y1, x0:x1], tec_aoi)


def test_get_ionex_value():
    '''
    Test IONEX TEC data interpolation
    '''

    # Lat/Lon coordinates over Chile
    lat, lon = -21.3, -67.4

    # 23:07 UTC time
    utc_sec = 23* 3600 +7 *60

    # Interpolation methods
    methods = ['nearest', 'linear2d', 'linear3d', 'linear3d']
    rotates = [False, False, False, True]
    values = [60.8, 58.90687978, 64.96605174, 65.15525905]

    # Get Ionex files
    tec_file = iono.get_ionex_filename(date_str,
                                       tec_dir=tec_dir,
                                       sol_code=sol_code,
                                       )
    # Perform comparison
    for method, rotate, value in zip(methods, rotates, values):
        tec_val = iono.get_ionex_value(tec_file,
                                       utc_sec,
                                       lat, lon,
                                       interp_method=method,
                                       rotate_tec_map=rotate,
                                       )
        assert np.allclose(tec_val, value, atol=1e-05, rtol=1e-05)


if __name__ == '__main__':
    test_read_ionex()
    test_get_ionex_value()
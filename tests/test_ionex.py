'''
Test IONEX functionalities: file reading and interpolation
'''

import numpy as np

from compass.utils import iono

def test_read_ionex(ionex_params):
    '''
    Test the reader for IONEX data

    Parameters
    ----------
    ionex_params: types.SimpleNamespace
        Variable containing IONEX parameters
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

    # Read IONEX tec_maps data - ignore mins, lats, and lons
    _, _, _, tec_maps = iono.read_ionex(ionex_params.tec_file)[:4]
    assert np.allclose(tec_maps[time_ind, y0:y1, x0:x1], tec_aoi)


def test_get_ionex_value(ionex_params):
    '''
    Test IONEX TEC data interpolation

    Parameters
    ----------
    ionex_params: types.SimpleNamespace
        Variable containing IONEX parameters
    '''

    # Lat/Lon coordinates over Chile
    lat, lon = -21.3, -67.4

    # 23:07 UTC time
    utc_sec = 23* 3600 +7 *60

    # Interpolation methods
    methods = ['nearest', 'linear2d', 'linear3d', 'linear3d']
    rotates = [False, False, False, True]
    values = [60.8, 58.90687978, 64.96605174, 65.15525905]

    # Perform comparison
    for method, rotate, value in zip(methods, rotates, values):
        tec_val = iono.get_ionex_value(ionex_params.tec_file,
                                       utc_sec, lat, lon,
                                       interp_method=method,
                                       rotate_tec_map=rotate,
                                       )
        assert np.allclose(tec_val, value, atol=1e-05, rtol=1e-05)


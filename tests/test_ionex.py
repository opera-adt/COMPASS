'''
Test IONEX functionalities: file reading and interpolation
'''
import os

import numpy as np
import pytest
from compass.utils import iono

@pytest.mark.vcr
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

@pytest.mark.vcr
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

@pytest.mark.vcr
def test_download_ionex(ionex_params):
    tec_file = iono.download_ionex(ionex_params.date_str,
                                   ionex_params.tec_dir,
                                   sol_code=ionex_params.sol_code)
    assert os.path.basename(tec_file) == 'jplg3190.15i'


@pytest.mark.vcr
def test_get_ionex_naming_format_old():
    # old IONEX filename format
    tec_url_old = iono.get_ionex_filename('20161115',
                                           sol_code='jpl',
                                           check_if_exists=True)
    print('tec_url_old:', tec_url_old)
    assert tec_url_old == ('https://cddis.nasa.gov/archive/gnss/products/'
                           'ionex/2016/320/jplg3200.16i.Z')

@pytest.mark.vcr
def test_get_ionex_naming_new():
    # new IONEX filename format
    tec_url_new = iono.get_ionex_filename('20241115',
                                           sol_code='jpl',
                                           check_if_exists=True)
    print('tec_url_new:', tec_url_new)
    assert tec_url_new == ('https://cddis.nasa.gov/archive/gnss/products/'
                           'ionex/2024/320/JPL0OPSFIN_20243200000_01D_02H_GIM.INX.gz')

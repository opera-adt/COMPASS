import multiprocessing as mp
import os
import pathlib
import types

import pytest
import requests
from s1reader.s1_orbit import check_internet_connection

from compass.utils import iono
from compass.utils.h5_helpers import DATA_PATH


def download_if_needed(local_path):
    '''
    Check if given path to file exists. Download if it from zenodo does not.

    Parameters
    ----------
    local_path: str
        Path to file
    '''
    # return if file is found
    if os.path.isfile(local_path):
        return

    check_internet_connection()

    dst_dir, file_name = os.path.split(local_path)

    # create destination directory if it does not exist
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)

    # download data
    dataset_url = 'https://zenodo.org/record/7668411/files/'
    target_url = f'{dataset_url}/{file_name}'
    with open(local_path, 'wb') as f:
        f.write(requests.get(target_url).content)


@pytest.fixture(scope="session")
def geocode_slc_params():
    '''
    Parameters to be used by geocode SLC unit test

    Returns
    -------
    test_params: SimpleNamespace
        SimpleNamespace containing geocode SLC unit test parameters
    '''
    test_params = types.SimpleNamespace()

    # burst ID and date of burst
    burst_id = 't064_135523_iw2'
    burst_date = '20221016'

    # get test working directory
    test_path = pathlib.Path(__file__).parent.resolve()

    # set other paths relative to working directory
    test_data_path = f'{test_path}/data'

    # paths for template and actual runconfig
    gslc_template_path = f'{test_data_path}/geo_cslc_s1_template.yaml'
    test_params.gslc_cfg_path = f'{test_data_path}/geo_cslc_s1.yaml'

    # read runconfig template, replace pieces, write to runconfig
    with open(gslc_template_path, 'r') as f_template, \
            open(test_params.gslc_cfg_path, 'w') as f_cfg:
        cfg = f_template.read().replace('@TEST_PATH@', str(test_path)).\
            replace('@DATA_PATH@', test_data_path).\
            replace('@BURST_ID@', burst_id)
        f_cfg.write(cfg)

    # files needed for geocode SLC unit test
    test_files = ['S1A_IW_SLC__1SDV_20221016T015043_20221016T015111_045461_056FC0_6681.zip',
                  'orbits/S1A_OPER_AUX_POEORB_OPOD_20221105T083813_V20221015T225942_20221017T005942.EOF',
                  'test_dem.tiff', 'test_burst_map.sqlite3',
                  '2022-10-16_0000_Rosamond-corner-reflectors.csv']
    test_files = [f'{test_data_path}/{test_file}' for test_file in test_files]

    # parallel download of test files (if necessary)
    with mp.Pool(len(test_files)) as pool:
        pool.map(download_if_needed, test_files)

    # path to file containing corner reflectors
    test_params.corner_coord_csv_path = test_files[-1]

    # path the output HDF5
    output_path = f'{test_path}/product/{burst_id}/{burst_date}'
    output_file_name = f'{burst_id}_{burst_date}.h5'
    test_params.output_hdf5_path = f'{output_path}/{output_file_name}'

    # path to groups and datasets in output HDF5
    test_params.data_group_path = DATA_PATH
    test_params.raster_path = f'{test_params.data_group_path}/VV'

    return test_params

@pytest.fixture(scope='session')
def ionex_params(download_data=True):
    '''
    Prepare IONEX data for unit test

    Parameters
    ----------
    download_data: bool
        Boolean flag allow to download TEC data
        for unit test if set to True

    Returns
    -------
    test_params : SimpleNamespace
        SimpleNamespace containing parameters needed for ionex unit test
    '''
    test_params = types.SimpleNamespace()

    # Set the path to fetch data for the test
    test_params.tec_dir = os.path.join(os.path.dirname(__file__), "data")
    test_params.date_str = '20151115'
    test_params.sol_code = 'jpl'

    # Create TEC directory
    os.makedirs(test_params.tec_dir, exist_ok=True)

    # Generate the TEC filename
    test_params.tec_file = iono.get_ionex_filename(test_params.date_str,
                                          tec_dir=test_params.tec_dir,
                                          sol_code=test_params.sol_code)

    # TODO figure out how to toggle download

    # If prep_mode=True, download data
    if download_data:
        if not os.path.isfile(test_params.tec_file):
            print(f'Download IONEX file at {test_params.date_str} from '
                  f'{test_params.sol_code} to {test_params.tec_dir}')
            test_params.tec_file = iono.download_ionex(test_params.date_str,
                                                       test_params.tec_dir,
                                                       sol_code=test_params.sol_code)

    return test_params

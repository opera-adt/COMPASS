from functools import partial
import multiprocessing as mp
import os
import pathlib
import types

import pytest
import requests
from s1reader.s1_orbit import check_internet_connection

from compass.utils import iono

def download_if_needed(local_path):
    # check if test inputs and reference files exists; download if not found.
    if os.path.isfile(local_path):
        return

    check_internet_connection()

    dataset_url = 'https://zenodo.org/record/7668410/files/'
    dst_dir, file_name = os.path.split(local_path)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)
    target_url = f'{dataset_url}/{file_name}'
    with open(local_path, 'wb') as f:
        f.write(requests.get(target_url).content)

@pytest.fixture(scope="session")
def unit_test_paths():
    test_paths = types.SimpleNamespace()

    burst_id = 't064_135523_iw2'
    b_date = '20221016'

    # get test working directory
    test_path = pathlib.Path(__file__).parent.resolve()

    # set other paths relative to working directory
    test_data_path = f'{test_path}/data'

    # paths for template and actual runconfig
    gslc_template_path = f'{test_data_path}/geo_cslc_s1_template.yaml'
    test_paths.gslc_cfg_path = f'{test_data_path}/geo_cslc_s1.yaml'

    # read runconfig template, replace pieces, write to runconfig
    with open(gslc_template_path, 'r') as f_template, \
            open(test_paths.gslc_cfg_path, 'w') as f_cfg:
        cfg = f_template.read().replace('@TEST_PATH@', str(test_path)).\
            replace('@DATA_PATH@', test_data_path).\
            replace('@BURST_ID@', burst_id)
        f_cfg.write(cfg)

    # check for files and download as needed
    test_files = ['S1A_IW_SLC__1SDV_20221016T015043_20221016T015111_045461_056FC0_6681.zip',
                  'orbits/S1A_OPER_AUX_POEORB_OPOD_20221105T083813_V20221015T225942_20221017T005942.EOF',
                  'test_dem.tiff', 'test_burst_map.sqlite3',
                  '2022-10-16_0000_Rosamond-corner-reflectors.csv']
    #for test_file in test_files:
    #    download_if_needed(f'{test_data_path}/{test_file}')
    pool = mp.Pool(len(test_files))
    _ = pool.map(download_if_needed, test_files)
    pool.close()
    pool.join()

    test_paths.corner_coord_csv_path = f'{test_data_path}/{test_files[-1]}'
    test_paths.output_hdf5 = f'{test_path}/product/{burst_id}/{b_date}/{burst_id}_{b_date}.h5'
    test_paths.grid_group_path = '/science/SENTINEL1/CSLC/grids'

    return test_paths

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
    tec_file: str
        Path to local or downloaded TEC file to
        use in the unit test
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

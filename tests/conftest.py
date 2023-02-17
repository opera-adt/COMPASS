import os
import pathlib
import requests
import types

from compass.utils import iono
import pytest
from s1reader.s1_orbit import check_internet_connection

def download_if_needed(local_path):
    # check if test inputs and reference files exists; download if not found.
    if os.path.isfile(local_path):
        return

    check_internet_connection()

    dataset_url = 'https://zenodo.org/record/6954753/files/'
    dst_dir, file_name = os.path.split(local_path)
    print(dst_dir, file_name)
    os.makedirs(dst_dir, exist_ok=True)
    target_url = f'{dataset_url}/{file_name}'
    with open(local_path, 'wb') as f:
        f.write(requests.get(target_url).content)

@pytest.fixture(scope="session")
def test_paths():
    test_paths = types.SimpleNamespace()

    burst_id = 't071_151200_iw1'
    b_date = '20200511'

    # get test working directory
    test_path = pathlib.Path(__file__).parent.resolve()

    # set other paths relative to working directory
    test_data_path = f'{test_path}/data'
    out_path = f'{test_path}/product'

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

    # output geocoded SLC paths
    test_paths.test_gslc = f'{out_path}/{burst_id}/{b_date}/{burst_id}_VV.slc'

    # reference geocoded SLC paths
    test_paths.ref_gslc = f'{test_data_path}/reference/ref_compass_gslc.slc'

    # check for files and download as needed
    test_files = ['S1A_IW_SLC__1SSV_20200511T135117_20200511T135144_032518_03C421_7768.zip',
                  'orbits/S1A_OPER_AUX_POEORB_OPOD_20210318T120818_V20200510T225942_20200512T005942.EOF',
                  'test_dem.tiff', 'reference/ref_compass_gslc.hdr',
                  'reference/ref_compass_gslc.slc']
    for test_file in test_files:
        download_if_needed(f'{test_data_path}/{test_file}')

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

import os
import pathlib
import pytest
import requests
import types

def download_if_needed(local_path):
    # check if test inputs and reference files exists; download if not found.
    if os.path.isfile(local_path):
        return

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

    b_id = 't71_151200_iw1'
    b_date = '20200511'

    # get test working directory
    test_path = pathlib.Path(__file__).parent.resolve()

    # set other paths relative to working directory
    test_data_path = f'{test_path}/data'
    out_path = f'{test_path}/gburst'

    # paths for template and actual runconfig
    gslc_template_path = f'{test_data_path}/geo_cslc_s1_template.yaml'
    test_paths.gslc_cfg_path = f'{test_data_path}/geo_cslc_s1.yaml'

    # read runconfig template, replace pieces, write to runconfig
    with open(gslc_template_path, 'r') as f_template, \
            open(test_paths.gslc_cfg_path, 'w') as f_cfg:
        cfg = f_template.read().replace('@TEST_PATH@', str(test_path)).\
            replace('@DATA_PATH@', test_data_path).\
            replace('@BURST_ID@', b_id)
        f_cfg.write(cfg)

    # output geocoded SLC paths
    test_paths.test_gslc = f'{out_path}/{b_id}/{b_date}/{b_id}_{b_date}_VV.slc'

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

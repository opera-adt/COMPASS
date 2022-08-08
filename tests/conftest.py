import pathlib
import pytest
import types

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

    # output and reference geocoded SLC paths
    test_paths.test_gslc = f'{out_path}/{b_id}/{b_date}/{b_id}_{b_date}_VV.slc'
    test_paths.ref_gslc = f'{test_path}/data/reference/ref_compass_gslc.slc'

    return test_paths

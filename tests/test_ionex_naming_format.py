import os
from compass.utils import iono
import pytest

@pytest.mark.vcr
def test_get_ionex_naming_format(ionex_params):
    # old IONEX filename format
    tec_file_old = iono.download_ionex('20161115',
                                   ionex_params.tec_dir,
                                   sol_code=ionex_params.sol_code)

    assert os.path.basename(tec_file_old) == 'jplg3200.16i'

    # new IONEX filename format
    tec_file_new = iono.download_ionex('20241115',
                                   ionex_params.tec_dir,
                                   sol_code=ionex_params.sol_code)
    assert os.path.basename(tec_file_new) == 'JPL0OPSFIN_20243200000_01D_02H_GIM.INX'

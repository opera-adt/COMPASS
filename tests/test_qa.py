import h5py
import numpy as np

from compass.utils.h5_helpers import QA_PATH

def test_qa_power_stats(geocode_slc_params):
    def _power_test(arr):
        return arr > 0.0

    def _phase_test(arr):
        return abs(arr) > np.pi

    # basic sanity checks of mean, min, and max
    with h5py.File(geocode_slc_params.output_hdf5_path, 'r') as h5_obj:
        for pwr_phase, test in {'power':_power_test,
                                'phase':_phase_test}.items():
            h5_stats_path = f'{QA_PATH}/statistics/data/VV/{pwr_phase}'
            stat_names = ['mean', 'min', 'max']
            for stat_name in stat_names:
                print(f'Testing: {stat_name} of {pwr_phase}')
                stat_val = h5_obj[f'{h5_stats_path}/{stat_name}'][()]
                assert test(stat_val)

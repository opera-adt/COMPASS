'''
Class to compute stats for geocoded raster and corrections
'''
import json

import h5py
import isce3
import numpy as np

from utils.h5_helpers import GRID_PATH, ROOT_PATH


class StatsCSLC:
    '''
    Class to compute stats for geocoded raster and corrections
    '''
    STAT_NAMES = ['mean', 'min', 'max', 'std']

    def __init__(self):
        self.stats_dict = {}


    def add_CSLC_raster_stats(self, path_h5, bursts):
        # compute and save CSLC raster(s) stats
        for b in bursts:
            pol = b.polarization
            qa_stat_items.append(pol)
            src_paths.append(f'{GRID_PATH}/{pol}')

        # compute stats and write to hdf5
        with h5py.File(path_h5, 'a') as h5_obj:
            for src_path, qa_stat_item in zip(src_paths, qa_stat_items):
                # get dataset and compute stats according to dtype
                ds = h5_obj[src_path]

                # compute stats for real and complex
                stat_obj = isce3.math.StatsRealImagFloat32(ds[()])

                self.stats_dict[qa_stat_item] = {}
                qa_stat_item = self.stats_dict[qa_stat_item]

                # write stats to HDF5
                for r_i, cstat_member in zip(['real', 'imag'],
                                             [stat_obj.real, stat_obj.imag]):
                    # create dict to store stat items
                    qa_stat_item[r_i] = {}

                    vals = [cstat_member.mean, cstat_member.min,
                            cstat_member.max, cstat_member.sample_stddev]
                    # write stats to HDF5
                    for val_name, val in zip(self.STAT_NAMES, vals):
                        qa_stat_item[r_i][val_name] = vals


    def add_correction_stats(self, path_h5):
        # compute and save corrections stats
        corrections = ['bistatic_delay', 'geometry_steering_doppler',
                       'azimuth_fm_rate_mismatch']
        for correction in corrections:
            src_paths.append(f'{ROOT_PATH}/corrections/{correction}')

        # compute stats and write to hdf5
        with h5py.File(path_h5, 'a') as h5_obj:
            self.stats_dict['corrections'] = {}
            corrections_dict = self.stats_dict['corrections']
            for src_path, correction_item in zip(src_paths, corrections):
                corrections_dict[correction_item] = {}
                correction_item_dict = corrections_dict[correction_item]

                # get dataset and compute stats according to dtype
                ds = h5_obj[src_path]
                stat_obj = isce3.math.StatsFloat32(ds[()].astype(np.float32))

                # write stats to HDF5
                vals = [stat_obj.mean, stat_obj.min, stat_obj.max]
                for val_name, val in zip(stat_names, vals):
                    correction_item_dict[val_name] = vals


    def raster_pixel_classification(self):
        pass


    def write_stats_to_json(self, filename):
        with open(filename, 'w') as f:
            json.dumps(self.stats_dict, f)

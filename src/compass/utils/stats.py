'''
Class to compute stats for geocoded raster and corrections
'''
import json

import h5py
import isce3
import numpy as np

from compass.utils.h5_helpers import GRID_PATH, ROOT_PATH


class StatsCSLC:
    '''
    Class to compute stats for geocoded raster and corrections
    '''
    stat_names = ['mean', 'min', 'max', 'std']

    def __init__(self):
        self.stats_dict = {}


    def compute_CSLC_raster_stats(self, cslc_h5py_root, bursts):
        '''
        Compute CSLC raster stats

        Parameters
        ----------
        cslc_h5py_root: h5py.File
            Root of CSLC HDF5
        bursts: list
            Bursts whose geocoded raster stats are to be computed
        '''
        for b in bursts:
            pol = b.polarization

            # get dataset and compute stats according to dtype
            src_path = f'{GRID_PATH}/{pol}'
            ds = cslc_h5py_root[src_path]

            # compute stats for real and complex
            stat_obj = isce3.math.StatsRealImagFloat32(ds[()])

            # create dict for current polarization
            self.stats_dict[pol] = {}
            pol_dict = self.stats_dict[pol]

            # write stats to HDF5
            for real_imag, cstat_member in zip(['real', 'imag'],
                                               [stat_obj.real, stat_obj.imag]):
                # create dict to store real/imaginary stat items
                pol_dict[real_imag] = {}

                vals = [cstat_member.mean, cstat_member.min,
                        cstat_member.max, cstat_member.sample_stddev]
                # write stats to HDF5
                for val_name, val in zip(self.stat_names, vals):
                    pol_dict[real_imag][val_name] = val


    def compute_correction_stats(self, cslc_h5py_root):
        '''
        Compute correction stats

        Parameters
        ----------
        cslc_h5py_root: h5py.File
            Root of CSLC HDF5
        '''
        # compute and save corrections stats
        corrections = ['bistatic_delay', 'geometry_steering_doppler',
                       'azimuth_fm_rate_mismatch']

        # compute stats and write to hdf5
        self.stats_dict['corrections'] = {}
        corrections_dict = self.stats_dict['corrections']
        for correction in corrections:
            # create dict for current correction
            corrections_dict[correction] = {}
            correction_dict = corrections_dict[correction]

            # get dataset and compute stats according to dtype
            src_path = f'{ROOT_PATH}/corrections/{correction}'
            ds = cslc_h5py_root[src_path]
            stat_obj = isce3.math.StatsFloat32(ds[()].astype(np.float32))

            # write stats to HDF5
            vals = [stat_obj.mean, stat_obj.min, stat_obj.max]
            for val_name, val in zip(self.stat_names, vals):
                correction_dict[val_name] = val


    def raster_pixel_classification(self):
        '''
        Place holder for classifier of geocoded pixel types
        '''
        pass


    def write_stats_to_json(self, file_path):
        '''
        Write computed stats in dict to JSON file

        Parameters
        ----------
        file_path: str
            JSON file to write stats to
        '''
        with open(file_path, 'w') as f:
            json.dump(self.stats_dict, f, indent=4)

'''
Class to compute stats for geocoded raster and corrections
'''
import json
from pathlib import Path

import isce3
import numpy as np

from compass.utils.h5_helpers import GRID_PATH, ROOT_PATH


def value_description_dict(val, desc):
    '''
    Convenience function that returns dict with description and value
    '''
    return {'value': val, 'description': desc}


class QualityAssuranceCSLC:
    '''
    Class to compute stats for geocoded raster and corrections
    '''
    stat_names = ['mean', 'min', 'max', 'std']
    stat_desc = 'Mean, minimum, maximum, and standard deviation of '

    def __init__(self):
        self.stats_dict = {}
        self.classification_count_dict = {}
        self.rfi_dict = {}
        self.is_safe_corrupt = False
        self.orbit_dict = {}


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

                # add description for stat items
                qa_item_desc = f'{real_imag} part of geoocoded SLC'

                vals = [cstat_member.mean, cstat_member.min,
                        cstat_member.max, cstat_member.sample_stddev]
                # write stats to HDF5
                for val_name, val in zip(self.stat_names, vals):
                    pol_dict[real_imag][val_name] = value_description_dict(
                        val, f'{val_name} of {qa_item_desc}')


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
                correction_dict[val_name] = value_description_dict(
                    val, f'{val_name} of {correction} correction')


    def raster_pixel_classification(self):
        '''
        Place holder for populating classification of geocoded pixel types
        '''
        self.classification_count_dict['topo'] = {}
        topo_dict = self.classification_count_dict['topo']
        topo_dict['percent_layover_pixels'] = value_description_dict(
            0, 'Percentage of output pixels labeled layover')
        topo_dict['percent_shadow_pixels'] = value_description_dict(
            0, 'Percentage of output pixels labeled shadow')
        topo_dict['percent_combined_pixels'] = value_description_dict(
            0, 'Percentage of output pixels labeled layover and shadow')
        self.classification_count_dict['percent_land_pixels'] = \
            value_description_dict(
                0, 'Percentage of output pixels labeld as land')
        self.classification_count_dict['percent_valid_pixels'] = \
            value_description_dict(
                0, 'Percentage of output pixels are valid')


    def populate_rfi_dict(self):
        '''
        Place holder for populating SAFE RFI information
        '''
        self.rfi_dict['is_rfi_info_available'] = value_description_dict(
            True, 'Whether or not RFI information is available')
        # Follow key/values only assigned if RFI info is avaiable
        self.rfi_dict['rfi_mitigation_performed'] = value_description_dict(
            True, 'Whether or not the RFI mitigation step was performed')
        self.rfi_dict['rfi_mitigation_domain'] = value_description_dict(
            True, 'Domain the RFI mitigation step was performed')
        self.rfi_dict['rfi_burst_report'] = value_description_dict(
            '', 'Burst RFI report')



    def set_orbit_type(self, cfg):
        '''
        Populate QA orbit information
        '''
        orbit_file_path = cfg.orbit_path[0]
        if 'RESORB' in orbit_file_path:
            orbit_type = 'restituted orbit file'
        if 'POEORB' in orbit_file_path:
            orbit_type = 'precise orbit file'
        self.orbit_dict['orbit_type'] = value_description_dict(
            orbit_type, 'Type of orbit file used in processing')


    def write_qa_dicts_to_json(self, file_path):
        '''
        Write computed stats in dict to JSON file

        Parameters
        ----------
        file_path: str
            JSON file to write stats to
        '''
        # combine all the dicts into one for output
        output_dict = {
            'raster_statistics': self.stats_dict,
            'pixel_classification_percentatges': self.classification_count_dict,
            'rfi_info': self.rfi_dict, 'orbit_info': self.orbit_dict}
        print(output_dict)

        # write combined dict to JSON
        with open(file_path, 'w') as f:
            json.dump(output_dict, f, indent=4)

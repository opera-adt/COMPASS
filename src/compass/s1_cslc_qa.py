'''
Class to compute stats for geocoded raster and corrections
'''
import json
from pathlib import Path

import isce3
import numpy as np

from compass.utils.h5_helpers import (DATA_PATH, QA_PATH, ROOT_PATH,
                                      add_dataset_and_attrs, Meta)


def value_description_dict(val, desc):
    '''
    Convenience function that returns dict with description and value
    '''
    return {'value': val, 'description': desc}


def _qa_items_to_h5_and_dict(h5_group, qa_dict, qa_items):
    '''
    Convenience function that write QA items to HDF5 group and QA dict
    '''
    # write items to HDF5 and dict
    for qa_item in qa_items:
        # write to HDF5 group RFI info
        add_dataset_and_attrs(h5_group, qa_item)

        # add items to RFI dict
        qa_dict[qa_item.name] = value_description_dict(qa_item.value,
                                                       qa_item.description)


class QualityAssuranceCSLC:
    '''
    Class to compute stats for geocoded raster and corrections
    '''
    stat_names = ['mean', 'min', 'max', 'std']

    def __init__(self):
        self.stats_dict = {}
        self.pixel_percentage_dict = {}
        self.rfi_dict = {}
        self.is_safe_corrupt = False
        self.orbit_dict = {}
        self.output_to_json = False


    def compute_CSLC_raster_stats(self, cslc_h5py_root, bursts):
        '''
        Compute CSLC raster stats. Stats written to HDF5 and saved to class
        dict for later JSON output

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
            pol_path = f'{DATA_PATH}/{pol}'
            pol_ds = cslc_h5py_root[pol_path]

            # compute stats for real and complex
            stat_obj = isce3.math.StatsRealImagFloat32(pol_ds[()])

            # create dict for current polarization
            self.stats_dict[pol] = {}
            pol_dict = self.stats_dict[pol]

            # write stats to HDF5
            for real_imag, cstat_member in zip(['real', 'imaginary'],
                                               [stat_obj.real, stat_obj.imag]):
                # create dict to store real/imaginary stat items
                pol_dict[real_imag] = {}

                # create HDF5 group for real/imaginary stats of current
                # polarization
                h5_stats_path = f'{QA_PATH}/statistics/data/{pol}/{real_imag}'
                stats_group = cslc_h5py_root.require_group(h5_stats_path)

                # add description for stat items
                suffix_qa_item_desc = f'{real_imag} part of geoocoded SLC'

                # build list of QA stat items for real_imag
                qa_items = []
                vals = [cstat_member.mean, cstat_member.min,
                        cstat_member.max, cstat_member.sample_stddev]
                for val_name, val in zip(self.stat_names, vals):
                    desc = f'{val_name} of {suffix_qa_item_desc}'
                    qa_items.append(Meta(val_name, val, desc))

                # save stats to dict and write to HDF5
                _qa_items_to_h5_and_dict(stats_group, pol_dict[real_imag],
                                      qa_items)


    def compute_static_layer_stats(self, cslc_h5py_root, rdr2geo_params):
        '''
        Compute correction stats. Stats written to HDF5 and saved to class dict
        for later JSON output

        Parameters
        ----------
        cslc_h5py_root: h5py.File
            Root of CSLC HDF5
        apply_tropo_corrections: bool
            Whether or not to compute troposhpere correction stats
        tropo_delay_type: str
            Type of troposphere delay. Any between 'dry', or 'wet', or
            'wet_dry' for the sum of wet and dry troposphere delays. Only used
            apply_tropo_corrections is true.
        '''
        # path to source group
        static_layer_path = f'{DATA_PATH}/static_layers'

        # Get the static layer to compute stats for
        static_layers_dict = {
            'x': rdr2geo_params.compute_longitude,
            'y': rdr2geo_params.compute_latitude,
            'z': rdr2geo_params.compute_height,
            'incidence': rdr2geo_params.compute_incidence_angle,
            'local_incidence': rdr2geo_params.compute_local_incidence_angle,
            'heading': rdr2geo_params.compute_azimuth_angle
        }
        static_layers = [key for key, val in static_layers_dict.items()
                         if val]

        self.compute_stats_from_float_hdf5_dataset(cslc_h5py_root,
                                                   static_layer_path,
                                                   'static_layers',
                                                   static_layers)


    def compute_correction_stats(self, cslc_h5py_root, apply_tropo_corrections,
                                 tropo_delay_type):
        '''
        Compute correction stats. Stats written to HDF5 and saved to class dict
        for later JSON output

        Parameters
        ----------
        cslc_h5py_root: h5py.File
            Root of CSLC HDF5
        apply_tropo_corrections: bool
            Whether or not to compute troposhpere correction stats
        tropo_delay_type: str
            Type of troposphere delay. Any between 'dry', or 'wet', or
            'wet_dry' for the sum of wet and dry troposphere delays. Only used
            apply_tropo_corrections is true.
        '''
        # path to source group
        corrections_src_path = f'{ROOT_PATH}/corrections'

        # names of datasets to compute stats for
        corrections = ['bistatic_delay', 'geometry_steering_doppler',
                       'azimuth_fm_rate_mismatch', 'los_ionospheric_delay',
                       'los_solid_earth_tides']

        # check if tropo corrections need to be computed and saved
        if apply_tropo_corrections:
            for delay_type in ['wet', 'dry']:
                if delay_type in tropo_delay_type:
                    corrections.append(f'{delay_type}_los_troposphere_delay')

        self.compute_stats_from_float_hdf5_dataset(cslc_h5py_root,
                                                   corrections_src_path,
                                                   'corrections', corrections)


    def compute_stats_from_float_hdf5_dataset(self, cslc_h5py_root,
                                              src_group_path, qa_group_name,
                                              qa_item_names):
        '''
        Compute correction stats for float-type, HDF5datasets. Stats written to
        HDF5 and saved to class dict for later JSON output

        Parameters
        ----------
        cslc_h5py_root: h5py.File
            Root of CSLC HDF5
        src_group_path: str
        qa_group_name: str
        qa_item_names: list[str]
        '''
        # init dict to save all QA item stats to
        self.stats_dict[qa_group_name] = {}
        qa_dict = self.stats_dict[qa_group_name]

        # compute stats and write to hdf5
        for qa_item_name in qa_item_names:
            # init dict for current QA item
            qa_dict[qa_item_name] = {}
            qa_item_dict = qa_dict[qa_item_name]

            # get dataset and compute stats according to dtype
            qa_item_path = f'{src_group_path}/{qa_item_name}'
            qa_item_ds = cslc_h5py_root[qa_item_path]

            # compute stats
            stat_obj = isce3.math.StatsFloat32(qa_item_ds[()].astype(np.float32))

            # create HDF5 group for stats of current QA item
            h5_stats_path = f'{QA_PATH}/statistics/{qa_group_name}/{qa_item_name}'
            qa_item_stats_group = cslc_h5py_root.require_group(h5_stats_path)

            # build list of QA stat items
            qa_items = []
            vals = [stat_obj.mean, stat_obj.min, stat_obj.max,
                    stat_obj.sample_stddev]
            for val_name, val in zip(self.stat_names, vals):
                desc = f'{val_name} of {qa_item_name}'
                qa_items.append(Meta(val_name, val, desc))

            # save stats to dict and write to HDF5
            _qa_items_to_h5_and_dict(qa_item_stats_group, qa_item_dict,
                                     qa_items)


    def shadow_pixel_classification(self, cslc_h5py_root):
        '''
        Place holder for populating classification of shadow layover pixels

        Parameters
        ----------
        cslc_h5py_root: h5py.File
            Root of CSLC HDF5
        '''
        pxl_qa_items = [
            Meta('percent_layover_pixels', 0,
                 'Percentage of output pixels labeled layover'),
            Meta('percent_shadow_pixels', 0,
                 'Percentage of output pixels labeled shadow'),
            Meta('percent_combined_pixels', 0,
                 'Percentage of output pixels labeled layover and shadow')
        ]

        # create HDF5 group for pixel classification info
        h5_pxl_path = f'{QA_PATH}/pixel_classification'
        pxl_group = cslc_h5py_root.require_group(h5_pxl_path)

        # write items to HDF5 and dict
        _qa_items_to_h5_and_dict(pxl_group, self.pixel_percentage_dict,
                                 pxl_qa_items)


    def valid_pixel_percentages(self, cslc_h5py_root):
        '''
        Place holder for populating classification of geocoded pixel types

        Parameters
        ----------
        cslc_h5py_root: h5py.File
            Root of CSLC HDF5
        '''
        pxl_qa_items = [
            Meta('percent_land_pixels', 0.0,
                 'Percentage of output pixels labeld as land'),
            Meta('percent_valid_pixels', 0.0,
                 'Percentage of output pixels are valid')
        ]

        # create HDF5 group for pixel classification info
        h5_pxl_path = f'{QA_PATH}/pixel_classification'
        pxl_group = cslc_h5py_root.require_group(h5_pxl_path)

        # write items to HDF5 and dict
        _qa_items_to_h5_and_dict(pxl_group, self.pixel_percentage_dict,
                                 pxl_qa_items)


    def populate_rfi_dict(self, cslc_h5py_root):
        '''
        Place holder for populating SAFE RFI information

        Parameters
        ----------
        cslc_h5py_root: h5py.File
            Root of CSLC HDF5
        '''
        rfi_qa_items = [
            Meta('is_rfi_info_available',True,
                 'Whether or not RFI information is available'),
            # Follow key/values only assigned if RFI info is avaiable
            Meta('rfi_mitigation_performed', True,
                 'Whether or not the RFI mitigation step was performed'),
            Meta('rfi_mitigation_domain', '',
                 'Domain the RFI mitigation step was performed'),
            Meta('rfi_burst_report', '', 'Burst RFI report')
        ]

        # create HDF5 group for RFI info
        h5_rfi_path = f'{QA_PATH}/rfi_information'
        rfi_group = cslc_h5py_root.require_group(h5_rfi_path)

        # write items to HDF5 and dict
        _qa_items_to_h5_and_dict(rfi_group, self.rfi_dict, rfi_qa_items)


    def set_orbit_type(self, cfg, cslc_h5py_root):
        '''
        Populate QA orbit information

        Parameters
        ----------
        cfg: dict
            Runconfig dict containing orbit path
        cslc_h5py_root: h5py.File
            Root of CSLC HDF5
        '''
        orbit_file_path = Path(cfg.orbit_path[0]).name
        if 'RESORB' in orbit_file_path:
            orbit_type = 'restituted orbit file'
        if 'POEORB' in orbit_file_path:
            orbit_type = 'precise orbit file'
        orbit_qa_items = [
            Meta('orbit_type', orbit_type,
                 'Type of orbit file used in processing')
        ]

        # create HDF5 group for orbit info
        h5_orbit_path = f'{QA_PATH}/orbit_information'
        orbit_group = cslc_h5py_root.require_group(h5_orbit_path)

        # write to HDF5 group orbit info
        _qa_items_to_h5_and_dict(orbit_group, self.orbit_dict, orbit_qa_items)


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
            'pixel_classification_percentatges': self.pixel_percentage_dict,
            'rfi_information': self.rfi_dict,
            'orbit_information': self.orbit_dict}

        # write combined dict to JSON
        with open(file_path, 'w') as f:
            json.dump(output_dict, f, indent=4)

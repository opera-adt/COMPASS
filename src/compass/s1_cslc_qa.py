'''
Class to compute stats for geocoded raster and corrections
'''
import datetime
import json
import os
from pathlib import Path
import time

import isce3
import numpy as np
from osgeo import ogr, osr, gdal
from scipy import ndimage

from compass.s1_rdr2geo import (file_name_los_east,
                                file_name_los_north,file_name_local_incidence,
                                file_name_x, file_name_y, file_name_z)
from compass.utils.h5_helpers import (DATA_PATH, METADATA_PATH, TIME_STR_FMT,
                                      QA_PATH, add_dataset_and_attrs, Meta)
from compass.utils.helpers import WORKFLOW_SCRIPTS_DIR

# determine the path to the world land GPKG file
LAND_GPKG_FILE = os.path.join(WORKFLOW_SCRIPTS_DIR, 'data',
                              'GSHHS_l_L1.shp.no_attrs.epsg3413_dissolved.gpkg')

def _compute_slc_array_stats(arr: np.ndarray, pwr_phase: str):
    # internal to function to compute min, max, mean, and std dev of power or
    # phase of SLC array. Default to phase stat computation.
    if pwr_phase == 'power':
        post_op_arr = np.abs(arr)**2
    else:
        post_op_arr = np.angle(arr)

    return [float(np_op(post_op_arr))
            for np_op in [np.nanmean, np.nanmin, np.nanmax,
                                      np.nanstd]]


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
            pol_arr = cslc_h5py_root[pol_path][()]

            # create dict for current polarization
            self.stats_dict[pol] = {}
            pol_dict = self.stats_dict[pol]

            # compute power or phase then write stats to HDF5 for CSLC
            for pwr_phase in ['power', 'phase']:
                # create dict to store real/imaginary stat items
                pol_dict[pwr_phase] = {}

                # create HDF5 group for power or phase stats of current
                # polarization
                h5_stats_path = f'{QA_PATH}/statistics/data/{pol}/{pwr_phase}'
                stats_group = cslc_h5py_root.require_group(h5_stats_path)

                # build list of QA stat items for pwr_phase
                qa_items = []
                vals = _compute_slc_array_stats(pol_arr, pwr_phase)
                for val_name, val in zip(self.stat_names, vals):
                    desc = f'{val_name} of {pwr_phase} of {pol} geocoded SLC'
                    qa_items.append(Meta(val_name, val, desc))

                # save stats to dict and write to HDF5
                _qa_items_to_h5_and_dict(stats_group, pol_dict[pwr_phase],
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
            Whether or not to compute troposphere correction stats
        tropo_delay_type: str
            Type of troposphere delay. Any between 'dry', or 'wet', or
            'wet_dry' for the sum of wet and dry troposphere delays. Only used
            apply_tropo_corrections is true.
        '''
        # path to source group
        static_layer_path = f'{DATA_PATH}'

        # Get the static layer to compute stats
        # Following dict tracks which static layers to generate
        # key: file name of static layer
        # value: bool flag from runconfig that determines if layer is computed
        static_layers_dict = {
            file_name_x: rdr2geo_params.compute_longitude,
            file_name_y: rdr2geo_params.compute_latitude,
            file_name_z: rdr2geo_params.compute_height,
            file_name_local_incidence: rdr2geo_params.compute_local_incidence_angle,
            file_name_los_east: rdr2geo_params.compute_ground_to_sat_east,
            file_name_los_north: rdr2geo_params.compute_ground_to_sat_north
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
            Whether or not to compute troposphere correction stats
        tropo_delay_type: str
            Type of troposphere delay. Any between 'dry', or 'wet', or
            'wet_dry' for the sum of wet and dry troposphere delays. Only used
            apply_tropo_corrections is true.
        '''
        # path to source group
        corrections_src_path = f'{METADATA_PATH}/processing_information/timing_corrections'

        # names of datasets to compute stats for
        corrections = ['bistatic_delay', 'geometry_steering_doppler',
                       'azimuth_fm_rate_mismatch', 'los_ionospheric_delay',
                       'los_solid_earth_tides', 'azimuth_solid_earth_tides']

        # check if tropo corrections need to be computed and saved
        if apply_tropo_corrections:
            for delay_type in ['wet', 'dry']:
                if delay_type in tropo_delay_type:
                    corrections.append(f'{delay_type}_los_troposphere_delay')

        self.compute_stats_from_float_hdf5_dataset(cslc_h5py_root,
                                                   corrections_src_path,
                                                   'timing_corrections', corrections)


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
        Populate classification of shadow layover pixels

        Parameters
        ----------
        cslc_h5py_root: h5py.File
            Root of CSLC HDF5
        '''

        percent_shadow, percent_layover, percent_combined =\
            self.compute_layover_shadow_pixel_percent(cslc_h5py_root)
        pxl_qa_items = [
            Meta('percent_layover_pixels', percent_layover,
                 'Percentage of output pixels labeled layover'),
            Meta('percent_shadow_pixels', percent_shadow,
                 'Percentage of output pixels labeled shadow'),
            Meta('percent_combined_pixels', percent_combined,
                 'Percentage of output pixels labeled layover and shadow')
        ]

        # create HDF5 group for pixel classification info
        h5_pxl_path = f'{QA_PATH}/pixel_classification'
        pxl_group = cslc_h5py_root.require_group(h5_pxl_path)

        # write items to HDF5 and dict
        _qa_items_to_h5_and_dict(pxl_group, self.pixel_percentage_dict,
                                 pxl_qa_items)


    def percent_land_and_valid_pixels(self, cslc_h5py_root, pol):
        '''
        Populate classification of geocoded pixel types

        Parameters
        ----------
        cslc_h5py_root: h5py.File
            Root of CSLC HDF5

        pol: str
            Polarization of the CSLC layer
        '''
        percent_land_pixels, percent_valid_pixels = \
            self.compute_valid_land_and_pixel_percents(cslc_h5py_root,
                                                       pol)
        pxl_qa_items = [
            Meta('percent_land_pixels', percent_land_pixels,
                 'Percentage of output pixels labeled as land'),
            Meta('percent_valid_pixels', percent_valid_pixels,
                 'Percentage of output pixels are valid')
        ]

        # create HDF5 group for pixel classification info
        h5_pxl_path = f'{QA_PATH}/pixel_classification'
        pxl_group = cslc_h5py_root.require_group(h5_pxl_path)

        # write items to HDF5 and dict
        _qa_items_to_h5_and_dict(pxl_group, self.pixel_percentage_dict,
                                 pxl_qa_items)


    def populate_rfi_dict(self, cslc_h5py_root, bursts):
        '''
        Place holder for populating SAFE RFI information

        Parameters
        ----------
        cslc_h5py_root: h5py.File
            Root of CSLC HDF5
        bursts: list[Sentinel1BurstSlc]
            List of burst SLC object with RFI info
        '''

        for burst in bursts:
            is_rfi_info_available = burst.burst_rfi_info is not None
            rfi_qa_items_pol = [Meta('is_rfi_info_available',
                                is_rfi_info_available,
                                'Whether or not RFI information is available')]

            if is_rfi_info_available:
                # Follow key/values only assigned if RFI info is available
                rfi_info_list = [
                    Meta('rfi_mitigation_performed',
                         burst.burst_rfi_info.rfi_mitigation_performed,
                         ('Activation strategy of RFI mitigation'
                          '["never", "BasedOnNoiseMeas", "always"]')),
                    Meta('rfi_mitigation_domain',
                         burst.burst_rfi_info.rfi_mitigation_domain,
                         'Domain the RFI mitigation step was performed')
                ]
                rfi_qa_items_pol += rfi_info_list

            # create HDF5 group for RFI info for current polarization
            h5_rfi_path = f'{QA_PATH}/rfi_information/{burst.polarization}'
            rfi_group = cslc_h5py_root.require_group(h5_rfi_path)

            # write items to HDF5 and dict
            _qa_items_to_h5_and_dict(rfi_group, self.rfi_dict, rfi_qa_items_pol)

            # Take care of the burst RFI report information
            if not is_rfi_info_available:
                return

            # Alias for readability
            rfi_burst_report = burst.burst_rfi_info.rfi_burst_report

            # Add the metadata of the burst RFI report
            rfi_burst_report_list = [
                Meta('swath',
                     rfi_burst_report['swath'],
                     'Swath of the burst'),
                Meta('azimuth_time',
                     datetime.datetime.strftime(rfi_burst_report['azimuthTime'],
                                                TIME_STR_FMT),
                    'Azimuth time of the burst report'),
                Meta('in_band_out_band_power_ratio',
                          rfi_burst_report['inBandOutBandPowerRatio'],
                          'Ratio between the in-band and out-of-band power of the burst')
            ]

            self.rfi_dict['rfi_burst_report'] = {}
            rfi_burst_report_group = rfi_group.require_group('rfi_burst_report')
            _qa_items_to_h5_and_dict(rfi_burst_report_group,
                                     self.rfi_dict['rfi_burst_report'],
                                     rfi_burst_report_list)

            # Take care of the time domain portion of the burst report
            if 'timeDomainRfiReport' in rfi_burst_report.keys():
                time_domain_report = rfi_burst_report['timeDomainRfiReport']
                burst_time_domain_report_item = [
                    Meta('percentage_affected_lines',
                         time_domain_report['percentageAffectedLines'],
                         'Percentage of level-0 lines affected by RFI.'),
                    Meta('avg_percentage_affected_samples',
                         time_domain_report['avgPercentageAffectedSamples'],
                         'Average percentage of affected level-0 samples in the lines containing RFI'),
                    Meta('max_percentage_affected_samples',
                         time_domain_report['maxPercentageAffectedSamples'],
                         'Maximum percentage of level-0 samples affected by RFI in the same line'),
                ]

                self.rfi_dict['rfi_burst_report']['time_domain_rfi_report'] = {}
                rfi_burst_report_time_domain_group =\
                    rfi_burst_report_group.require_group('time_domain_rfi_report')
                _qa_items_to_h5_and_dict(rfi_burst_report_time_domain_group,
                                         self.rfi_dict['rfi_burst_report']['time_domain_rfi_report'],
                                         burst_time_domain_report_item)

            # Take care of the frequency time domain portion of the burst report
            if 'frequencyDomainRfiBurstReport' in rfi_burst_report.keys():
                freq_domain_report = rfi_burst_report['frequencyDomainRfiBurstReport']
                burst_freq_domain_report_item = [
                    Meta('num_sub_blocks',
                         freq_domain_report['numSubBlocks'],
                         'Number of sub-blocks in the current burst'),
                    Meta('sub_block_size',
                         freq_domain_report['subBlockSize'],
                         'Number of lines in each sub-block'),
                    Meta('percentage_blocks_persistent_rfi',
                         freq_domain_report['percentageBlocksPersistentRfi'],
                         ('Percentage of processing blocks affected by persistent RFI. '
                          'In this case the RFI detection is performed on the mean PSD of '
                          'each processing block')),
                    Meta('max_percentage_bw_affected_persistent_rfi',
                         freq_domain_report['maxPercentageBWAffectedPersistentRfi'],
                         ('Max percentage bandwidth affected by '
                          'persistent RFI in a single processing block.'))
                ]

                self.rfi_dict['rfi_burst_report']['frequency_domain_rfi_report'] = {}
                rfi_burst_report_freq_domain_group = rfi_burst_report_group.require_group('frequency_domain_rfi_report')
                _qa_items_to_h5_and_dict(rfi_burst_report_freq_domain_group,
                                        self.rfi_dict['rfi_burst_report']['frequency_domain_rfi_report'],
                                        burst_freq_domain_report_item)

                # Take care of isolated RFI report inside frequency burst RFI report
                isolated_rfi_report = freq_domain_report['isolatedRfiReport']
                isolated_report_item = [
                    Meta('percentage_affected_lines',
                         isolated_rfi_report['percentageAffectedLines'],
                         'Percentage of level-0 lines affected by isolated RFI'),
                    Meta('max_percentage_affected_bw',
                         isolated_rfi_report['maxPercentageAffectedBW'],
                         'Max. percentage of bandwidth affected by isolated RFI in a single line')
                ]

                self.rfi_dict['rfi_burst_report']['time_domain_rfi_report']['isolated_rfi_report'] = {}
                isolated_rfi_report_group = rfi_burst_report_freq_domain_group.require_group('isolated_rfi_report')
                _qa_items_to_h5_and_dict(isolated_rfi_report_group,
                                        self.rfi_dict['rfi_burst_report']['time_domain_rfi_report']['isolated_rfi_report'],
                                        isolated_report_item)


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
                 'Type of orbit file used for processing. '
                 'RESORB: restituted orbit ephemeris or POEORB: precise orbit ephemeris')
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


    def compute_valid_land_and_pixel_percents(self, cslc_h5py_root, pol):
        '''
        Compute the percentage of valid pixels on land area

        Parameters
        ----------
        cslc_h5py_path: h5py.File
            Root of the CSLC-S1 HDF5 product
        pol: str
            Polarization of CSLC layer to
            compute the valid pixel area

        Returns
        -------
        percent_valid_land_px: float
            Percentage of valid pixels on land
            in the geocoded burst area
        percent_valid_px: float
            Percentage of invalid pixels
            in the geocoded burst area
        '''
        # extract the geogrid information
        epsg_cslc = int(cslc_h5py_root[f'{DATA_PATH}/projection'][()])

        x_spacing = float(cslc_h5py_root[f'{DATA_PATH}/x_spacing'][()])
        y_spacing = float(cslc_h5py_root[f'{DATA_PATH}/y_spacing'][()])

        x0 = list(cslc_h5py_root[f'{DATA_PATH}/x_coordinates'][()])[0] - x_spacing / 2
        y0 = list(cslc_h5py_root[f'{DATA_PATH}/y_coordinates'][()])[0] - y_spacing / 2

        cslc_array = np.array(cslc_h5py_root[f'{DATA_PATH}/{pol}'])

        height_cslc, width_cslc = cslc_array.shape

        mask_land = _get_land_mask(epsg_cslc,
                                   (x0, x_spacing, 0, y0, 0, y_spacing),
                                   (height_cslc, width_cslc))

        mask_geocoded_burst = _get_valid_pixel_mask(cslc_array)

        mask_valid_inside_burst = mask_geocoded_burst & ~np.isnan(cslc_array)

        mask_valid_land_pixel = mask_geocoded_burst & mask_land

        n_unmasked_pxls = mask_geocoded_burst.sum()
        percent_valid_land_px = mask_valid_land_pixel.sum() / n_unmasked_pxls * 100
        percent_valid_px = mask_valid_inside_burst.sum() / n_unmasked_pxls * 100

        return percent_valid_land_px, percent_valid_px


    def compute_layover_shadow_pixel_percent(self, cslc_h5py_root):
        '''
        Compute the percentage of layover, shadow, and
        layover/shadow pixels in the geocoded burst area

        Parameters
        ----------
        cslc_h5py_path: h5py.File
            Root of the CSLC-S1 HDF5 product

        Returns
        -------
        percent_shadow: float
            Percentage of the shadow pixels
            in the geocoded burst area
        percent_layover: float
            Percentage of the layover pixels
            in the geocoded burst area
        percent_combined: float
            Percentage of the shadow and layover pixels
            in the geocoded burst area
        '''
        layover_shadow_mask_array = cslc_h5py_root[f'{DATA_PATH}/layover_shadow_mask'][()]

        mask_geocoded_burst = layover_shadow_mask_array != 127

        n_unmasked_pxls = mask_geocoded_burst.sum()

        mask_shadow_inside_burst = mask_geocoded_burst & (layover_shadow_mask_array == 1)
        percent_shadow = mask_shadow_inside_burst.sum() / n_unmasked_pxls * 100

        mask_layover_inside_burst = mask_geocoded_burst & (layover_shadow_mask_array == 2)
        percent_layover = mask_layover_inside_burst.sum() / n_unmasked_pxls * 100

        mask_combined_inside_burst = mask_geocoded_burst & (layover_shadow_mask_array == 3)
        percent_combined = mask_combined_inside_burst.sum() / n_unmasked_pxls * 100

        return percent_shadow, percent_layover, percent_combined


def _get_valid_pixel_mask(arr_cslc):
    '''
    Get the binary index of the pixels in the geocoded burst area

    Parameters
    ----------
    arr_cslc: np.ndarray
        CSLC layer to compute the geocoded burst area

    Returns
    -------
    valid_pixel_index: np.ndarray
        binary index that identifies the pixels in the
        geocoded burst area
    '''
    mask_nan = np.isnan(arr_cslc)
    labeled_arr, _ = ndimage.label(mask_nan)

    labels_along_edges = np.concatenate((labeled_arr[0, :],
                                         labeled_arr[-1, :],
                                         labeled_arr[:, 0],
                                         labeled_arr[:, -1]))

    # Filter out the valid pixels that touches the edges
    labels_along_edges = labels_along_edges[labels_along_edges != 0]

    labels_edge_list = list(set(labels_along_edges))


    # Initial binary index array. Filled with `True`
    valid_pixel_index = np.full(labeled_arr.shape, True)

    # Get rid of the NaN area whose label is detected along the array's edges
    for labels_edge in labels_edge_list:
        valid_pixel_index[labeled_arr == labels_edge] = False

    return valid_pixel_index


def _get_land_mask(epsg_cslc: int, geotransform: tuple, shape_mask: tuple):
    '''
    Get the land mask within the CSLC bounding box

    Parameters
    ----------
    epsg_cslc: int
        EPSG code of the CSLC layer
    geotransform: tuple
        Geotransform vector of the CSLC layer
    shape_mask: tuple
        Shape of the raster as numpy array

    Returns
    -------
    mask_land: np.ndarray
        Raster Mask for land area. `1` is land, `0` otherwise
    '''
    # Extract the land polygon
    ds_land = ogr.Open(LAND_GPKG_FILE, 0)
    layer_land = ds_land.GetLayer()
    feature = layer_land.GetNextFeature()
    land_polygon = feature.GetGeometryRef()

    # extract the EPSG of the land polygon GPKG
    srs_gpkg = layer_land.GetSpatialRef()
    land_epsg = int(srs_gpkg.GetAuthorityCode(None))

    # Compute and create the bounding box
    xmin = geotransform[0]
    ymin = geotransform[3] + geotransform[5] * shape_mask[0]
    xmax = geotransform[0] + geotransform[1] * shape_mask[1]
    ymax = geotransform[3]
    bbox_cslc = ogr.Geometry(ogr.wkbPolygon)
    ring_cslc = ogr.Geometry(ogr.wkbLinearRing)
    ring_cslc.AddPoint(xmin, ymin)
    ring_cslc.AddPoint(xmax, ymin)
    ring_cslc.AddPoint(xmax, ymax)
    ring_cslc.AddPoint(xmin, ymax)
    ring_cslc.AddPoint(xmin, ymin)
    bbox_cslc.AddGeometry(ring_cslc)

    # Define the SRS for CSLC and land polygon
    srs_cslc = osr.SpatialReference()
    srs_cslc.ImportFromEPSG(epsg_cslc)

    srs_land = osr.SpatialReference()
    srs_land.ImportFromEPSG(land_epsg)

    # Reproject the bounding box (in CSLC EPSG) to land polygon's EPSG
    transformer_cslc_to_land = osr.CoordinateTransformation(srs_cslc, srs_land)
    bbox_cslc.Transform(transformer_cslc_to_land)

    # Return a numpy array full of `False` when there is no intersection
    if not bbox_cslc.Intersects(land_polygon):
        return np.full(shape_mask, False)

    # Compute the intersection and reproject the result back to CSLC's EPSG
    intersection_land = bbox_cslc.Intersection(land_polygon)
    transformer_land_to_cslc = osr.CoordinateTransformation(srs_land, srs_cslc)
    intersection_land.Transform(transformer_land_to_cslc)

    # Build up a vector layer, and add a feature that has `intersection_land`` as geometry
    drv_intersection_polygon = ogr.GetDriverByName('Memory')
    ds_intersection_polygon = drv_intersection_polygon.CreateDataSource(str(time.time_ns))
    layer_intersection = ds_intersection_polygon.CreateLayer('layer_intersection',
                                                             srs_cslc,
                                                             ogr.wkbPolygon)
    feature_defn = layer_intersection.GetLayerDefn()
    feature = ogr.Feature(feature_defn)
    feature.SetGeometry(intersection_land)
    layer_intersection.CreateFeature(feature)

    # Prepare for output layer for the rasterization
    drv_raster_out = gdal.GetDriverByName('MEM')
    rasterized_land = drv_raster_out.Create(str(time.time_ns),
                                            shape_mask[1], shape_mask[0],
                                            1, gdal.GDT_Byte)
    rasterized_land.SetGeoTransform(geotransform)
    rasterized_land.SetProjection(srs_cslc.ExportToWkt())

    gdal.RasterizeLayer(rasterized_land, [1], layer_intersection)

    mask_land = rasterized_land.ReadAsArray()

    return mask_land

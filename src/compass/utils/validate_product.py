#!/usr/bin/env python
'''
Collection of functions to compare 2 CSLC HDF5 contents and metadata
'''
import argparse
import os

import h5py
import numpy as np
from osgeo import gdal


DATA_ROOT = 'science/SENTINEL1'


def cmd_line_parser():
    """
    Command line parser
    """
    parser = argparse.ArgumentParser(
        description="""Validate reference and generated (secondary) S1 CSLC products""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-r', '--ref-product', type=str, dest='ref_product',
        help='Reference CSLC or static layer product (i.e., golden dataset)')
    parser.add_argument(
        '-s', '--sec-product', type=str, dest='sec_product',
        help='Secondary CSLC or static layer product to compare with reference')
    parser.add_argument('-p', '--product-type', type=str, dest='product_type',
                        choices=['CSLC', 'static_layer'],
                        default='CSLC', help='Type of file to be validated')
    return parser.parse_args()


def _grid_info_retrieve(path_h5, dataset_names, is_static_layer):
    """
    Extract names of found datasets, geotransform array, and projection from
    grid group of given HDF5

    Parameters
    ----------
    path_h5: str
        File path to CSLC or static layer HDF5 product
    dataset_names: list[str]
        List of names in grid group
    is_static_layer: bool
        Whether or not path_h5 is static layer product

    Returns
    -------
    set
        Set of names of datasets found in grid group
    geotransform: np.array
        Array holding x/y start, x/y spacing, and geogrid dimensions
    proj: str
        Map projection of the raster in WKT
    """
    grid_path = f'{DATA_ROOT}/CSLC/grids'
    if is_static_layer:
        grid_path += '/static_layers'

    # Extract existing dataset names with h5py
    with h5py.File(path_h5) as h:
        datasets_found = [ds_name for ds_name in dataset_names
                if ds_name in h[grid_path]]

    ds_name = datasets_found[0]

    # Extract some info from reference/secondary products with GDAL
    h5_gdal_path = f'NETCDF:{path_h5}://{grid_path}/{ds_name}'
    dataset = gdal.Open(h5_gdal_path, gdal.GA_ReadOnly)
    geotransform = dataset.GetGeoTransform()
    proj = dataset.GetProjection()

    return set(datasets_found), geotransform, proj


def compare_products(file_ref, file_sec, product_type):
    '''
    Compare a reference and a newly generated
    (i.e., secondary) CSLC or static layer product

    Parameters
    ----------
    file_ref: str
        File path to reference product (golden dataset)
    file_sec: str
        File path to generated product to use for comparison
    product_type: str
        Product type of CSLC or static_layer
    '''

    # Check if file paths exits
    if not os.path.exists(file_ref):
        print(f'ERROR reference {product_type} product not found: {file_ref}')
        return

    if not os.path.exists(file_sec):
        print(f'ERROR secondary {product_type} product not found: {file_sec}')
        return

    # Extract some info from reference/secondary products
    dataset_names = ['VV', 'VH', 'HH', 'HV'] if product_type == 'CSLC' else \
        ['x', 'y', 'z', 'incidence', 'local_incidence', 'heading',
         'layover_shadow_mask']
    is_static_layer = product_type == 'static_layer'
    items_ref, geotransform_ref, proj_ref = \
        _grid_info_retrieve(file_ref, dataset_names, is_static_layer)
    items_sec, geotransform_sec, proj_sec = \
        _grid_info_retrieve(file_sec, dataset_names, is_static_layer)

    # Intersect grid items found
    set_ref_minus_sec = items_ref - items_sec
    set_sec_minus_ref = items_sec - items_ref

    err_str = "Grid items do not match.\n"
    if set_ref_minus_sec:
        err_str += \
            f'\nReference {product_type} extra entries: {set_ref_minus_sec}'
    if set_sec_minus_ref:
        err_str += \
            f'\nSecondary {product_type} extra entries: {set_sec_minus_ref}'

    # Check if metadata key differ
    if set_ref_minus_sec or set_sec_minus_ref:
        print(err_str)
        return

    print(f'Comparing {product_type} projection ...')
    if not proj_ref == proj_sec:
        print(f'ERROR projection in reference {proj_ref} differs '
              f'from projection in secondary {proj_sec}')
        return

    print('Comparing geo transform arrays ...')
    if not np.array_equal(geotransform_ref, geotransform_sec):
        print(f'ERROR Reference geo transform array {geotransform_ref} differs'
              f'from secondary geo transform array {geotransform_sec}')
        return

    print('Comparing raster arrays...')
    compare_rasters = _compare_complex_slc_rasters if product_type == 'CSLC' \
        else _compare_static_layer_rasters
    compare_rasters(file_ref, file_sec, items_ref)


def _compare_static_layer_rasters(file_ref, file_sec, static_layer_items):
    """
    Compare reference and secondary static layer rasters for given static layer
    items

    Parameters
    ----------
    file_ref: str
        File path to reference product (golden dataset)
    file_sec: str
        File path to generated product to use for comparison
    static_layer_items: list[str]
        List of names of static layers to compare
    """
    grid_path = f'{DATA_ROOT}/CSLC/grids/static_layers'
    with h5py.File(file_ref, 'r') as h_ref, h5py.File(file_sec, 'r') as h_sec:
        for static_layer_item in static_layer_items:
            if static_layer_item == 'layover_shadow_mask':
                continue

            # Retrieve static layer raster from ref and sec HDF5
            slc_ref = h_ref[f'{grid_path}/{static_layer_item}']
            slc_sec = h_sec[f'{grid_path}/{static_layer_item}']

            # Compute total number of pixels different from nan from ref and sec
            ref_nan = np.isnan(slc_ref)
            sec_nan = np.isnan(slc_sec)

            # Check that the pixels is the same
            np.testing.assert_array_equal(ref_nan, sec_nan)

            # Compute absolute difference between real and imaginary
            ma_ref = np.ma.masked_array(slc_ref, mask=ref_nan)
            ma_sec = np.ma.masked_array(slc_sec, mask=sec_nan)
            ref_sec_diff = np.abs((ma_ref - ma_sec) / ma_sec)

            # Count the number of pixels in real and imaginary part above threshold
            pixel_diff_threshold = 1e-5
            failed_pixels = np.count_nonzero(
                ref_sec_diff > pixel_diff_threshold)

            # Compute percentage of pixels above threshold
            tot_pixels_ref = np.count_nonzero(ref_nan)
            percentage_fail = failed_pixels / tot_pixels_ref

            # Check that percentage of pixels above threshold is lower than 0.1 %
            total_fail_threshold = 0.001
            assert percentage_fail < total_fail_threshold, \
                f'{static_layer_item} exceeded {total_fail_threshold * 100} ' \
                '% of pixels where reference and secondary differed by more than ' \
                f'{pixel_diff_threshold}.'


def _compare_complex_slc_rasters(file_ref, file_sec, pols):
    """
    Compare reference and secondary complex rasters for given polarizations

    Parameters
    ----------
    file_ref: str
        File path to reference product (golden dataset)
    file_sec: str
        File path to generated product to use for comparison
    pols: list[str]
        List of polarizations of rasters to compare
    """
    grid_path = f'{DATA_ROOT}/CSLC/grids'
    with h5py.File(file_ref, 'r') as h_ref, h5py.File(file_sec, 'r') as h_sec:
        for pol in pols:
            # Retrieve SLC raster from ref and sec HDF5
            slc_ref = h_ref[f'{grid_path}/{pol}']
            slc_sec = h_sec[f'{grid_path}/{pol}']

            # Compute total number of pixels different from nan from ref and sec
            ref_nan = np.isnan(slc_ref)
            sec_nan = np.isnan(slc_sec)

            # Check that the pixels is the same
            np.testing.assert_array_equal(ref_nan, sec_nan)

            # Compute absolute difference between real and imaginary
            ma_slc_ref = np.ma.masked_array(slc_ref, mask=ref_nan)
            ma_slc_sec = np.ma.masked_array(slc_sec, mask=sec_nan)
            diff_real = \
                np.abs((ma_slc_ref.real - ma_slc_sec.real) / ma_slc_sec.real)
            diff_imag = \
                np.abs((ma_slc_ref.imag - ma_slc_sec.imag) / ma_slc_sec.imag)

            # Count the number of pixels in real and imaginary part above threshold
            pixel_diff_threshold = 1e-5
            failed_pixels_real = np.count_nonzero(diff_real > pixel_diff_threshold)
            failed_pixels_imag = np.count_nonzero(diff_imag > pixel_diff_threshold)

            # Compute percentage of pixels in real and imaginary part above threshold
            tot_pixels_ref = np.count_nonzero(ref_nan)
            percentage_real = failed_pixels_real / tot_pixels_ref
            percentage_imag = failed_pixels_imag / tot_pixels_ref

            # Check that percentage of pixels above threshold is lower than 0.1 %
            total_fail_threshold = 0.001
            fails = []
            if percentage_real >= total_fail_threshold:
                fails.append('real')
            if percentage_imag >= total_fail_threshold:
                fails.append('imaginary')

            # Format fails. join() doesn't affect emtpy lists
            fails = ', '.join(fails)
            assert len(fails) == 0, f'{fails} exceeded {total_fail_threshold * 100} ' \
                '% of pixels where reference and secondary differed by more than ' \
                f'{pixel_diff_threshold} in polarization {pol}.'


def _get_group_item_paths(h5py_group):
    '''
    Get paths for all datasets and groups nested within a h5py.Group

    Parameters
    ----------
    h5py_group: h5py.Group
        Group object where paths to objects within are to be retrieved.

    Returns
    -------
    paths: list[str]
        Paths of all items in given h5py.Group
    '''
    paths = []
    h5py_group.visit(lambda path: paths.append(path))
    return paths


def compare_cslc_metadata(file_ref, file_sec):
    '''
    Compare reference and generated CSLC metadata
    Parameters
    ----------
    file_ref: str
        File path to reference metadata file (golden dataset)
    file_sec: str
        File path to secondary metadata file to use for comparison
    '''

    # Check if metadata files exists
    if not os.path.exists(file_ref):
        print(f'ERROR reference CSLC metadata not found: {file_ref}')
        return

    if not os.path.exists(file_sec):
        print(f'ERROR CSLC metadata not found: {file_sec}')
        return

    # Get metadata keys
    with h5py.File(file_ref, 'r') as h_ref, h5py.File(file_sec, 'r') as h_sec:
        metadata_ref = set(_get_group_item_paths(h_ref[DATA_ROOT]))
        metadata_sec = set(_get_group_item_paths(h_sec[DATA_ROOT]))

    # Intersect metadata keys
    set_ref_minus_sec = metadata_ref - metadata_sec
    set_sec_minus_ref = metadata_sec - metadata_ref

    err_str = "Metadata keys do not match.\n"
    if set_ref_minus_sec:
        err_str += f'\nReference CSLC metadata extra entries: {set_ref_minus_sec}'
    if set_sec_minus_ref:
        err_str += f'\nSecondary CSLC metadata extra entries: {set_sec_minus_ref}'

    # Check if metadata key differ
    assert not set_ref_minus_sec or not set_sec_minus_ref, err_str


def main():
    '''Entrypoint of the script'''
    cmd = cmd_line_parser()

    # Check CSLC products
    compare_products(cmd.ref_product, cmd.sec_product, cmd.product_type)
    print('All CSLC product checks have passed')

    # Check CSLC metadata
    compare_cslc_metadata(cmd.ref_product, cmd.sec_product)
    print('All CSLC metadata checks have passed')


if __name__ == '__main__':
    main()

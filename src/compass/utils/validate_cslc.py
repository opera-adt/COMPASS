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
    parser = argparse.ArgumentParser(description="""Validate
                                     reference and generated (secondary) S1 CSLC products""",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--ref-product', type=str, dest='ref_product',
                        help='Reference CSLC product (i.e., golden dataset)')
    parser.add_argument('-s', '--sec-product', type=str, dest='sec_product',
                        help='Secondary CSLC product to compare with reference')
    return parser.parse_args()


def _gdal_nfo_retrieve(path_h5):
    """
    Extract polarization, geotransform array, projection, and SLC from
    given HDF5
    Parameters
    ----------
    path_h5: str
        File path to CSLC HDF5 product
    Returns
    -------
    pol: str
        Polarization of CSLC product
    geotransform: np.array
        Array holding x/y start, x/y spacing, and geogrid dimensions
    proj: str
        EPSG projection of geogrid
    slc: np.array
        Array holding geocoded complex backscatter
    """
    grid_path = f'{DATA_ROOT}/CSLC/grids'

    # Extract polarization with h5py
    with h5py.File(path_h5) as h:
        for pol in ['VV', 'VH', 'HH', 'HV']:
            if pol in h[grid_path]:
                break

    # Extract some info from reference/secondary CSLC products with GDAL
    h5_gdal_path = f'NETCDF:{path_h5}://{grid_path}/{pol}'
    dataset = gdal.Open(h5_gdal_path, gdal.GA_ReadOnly)
    geotransform = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    slc = dataset.GetRasterBand(1).ReadAsArray()

    return pol, geotransform, proj, slc


def compare_cslc_products(file_ref, file_sec):
    '''
    Compare a reference and a newly generated
    (i.e., secondary) CSLC product
    Parameters
    ----------
    file_ref: str
        File path to reference CSLC product (golden dataset)
    file_sec: str
        File path to generated CSLC product to use for comparison
    '''

    # Check if file paths exits
    if not os.path.exists(file_ref):
        print(f'ERROR reference CSLC product not found: {file_ref}')
        return

    if not os.path.exists(file_sec):
        print(f'ERROR secondary CSLC product not found: {file_sec}')
        return

    # Extract some info from reference/secondary CSLC products
    pol_ref, geotransform_ref, proj_ref, slc_ref = _gdal_nfo_retrieve(file_ref)
    pol_sec, geotransform_sec, proj_sec, slc_sec = _gdal_nfo_retrieve(file_sec)

    # Compare number of bands
    print('Comparing CSLC number of bands ...')
    if not pol_ref == pol_sec:
        print(f'ERROR Polarization in reference CSLC {pol_ref} differs '
              f'from polarization {pol_sec} in secondary CSLC')
        return

    print('Comparing CSLC projection ...')
    if not proj_ref == proj_sec:
        print(f'ERROR projection in reference CSLC {proj_ref} differs '
              f'from projection in secondary CSLC {proj_sec}')

    print('Comparing geo transform arrays ...')
    if not np.array_equal(geotransform_ref, geotransform_sec):
        print(f'ERROR Reference geo transform array {geotransform_ref} differs'
              f'from secondary CSLC geo transform array {geotransform_sec}')
        return

    # Compute total number of pixels different from nan from ref and sec
    ref_nan = np.isnan(slc_ref)
    sec_nan = np.isnan(slc_sec)

    # Check that the pixels is the same
    np.testing.assert_array_equal(ref_nan, sec_nan)

    # Compute absolute difference between real and imaginary
    ma_slc_ref = np.ma.masked_array(slc_ref, mask=ref_nan)
    ma_slc_sec = np.ma.masked_array(slc_sec, mask=sec_nan)
    diff_real = np.abs((ma_slc_ref.real - ma_slc_sec.real) / ma_slc_sec.real)
    diff_imag = np.abs((ma_slc_ref.imag - ma_slc_sec.imag) / ma_slc_sec.imag)

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
        f'{pixel_diff_threshold}.'


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
    compare_cslc_products(cmd.ref_product, cmd.sec_product)
    print('All CSLC product checks have passed')

    # Check CSLC metadata
    compare_cslc_metadata(cmd.ref_product, cmd.sec_product)
    print('All CSLC metadata checks have passed')


if __name__ == '__main__':
    main()

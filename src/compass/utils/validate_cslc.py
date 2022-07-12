import argparse
import json
import os

import numpy as np
from osgeo import gdal


def cmd_line_parser():
    """
    Command line parser
    """

    parser = argparse.ArgumentParser(description="""
                                     Validate reference and generated (secondary) S1 CSLC products""",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--ref-product', type=str, dest='ref_product',
                        help='Reference CSLC product (i.e., golden dataset)')
    parser.add_argument('-mr', '--ref-metadata', type=str, dest='ref_metadata',
                        help='Reference CSLC metadata file (i.e., golden metadata)')
    parser.add_argument('-s', '--sec-product', type=str, dest='sec_product',
                        help='Secondary CSLC product to compare with reference')
    parser.add_argument('-ms', '--sec-metadata', type=str, dest='sec_metadata',
                        help='Secondary CSLC metadata to compare with reference metadata')
    return parser.parse_args()


def compare_cslc_products(file_ref, file_sec):
    '''
    Compare a reference and a newly generated
    (i.e., secondary) CSLC product

    Parameters
    ----------
    file1: str
        File path to reference CSLC product (golden dataset)
    file2: str
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
    dataset_ref = gdal.Open(file_ref, gdal.GA_ReadOnly)
    geotransform_ref = dataset_ref.GetGeoTransform()
    nbands_ref = dataset_ref.RasterCount

    dataset_sec = gdal.Open(file_sec, gdal.GA_ReadOnly)
    geotransform_sec = dataset_sec.GetGeoTransform()
    nbands_sec = dataset_sec.RasterCount

    # Compare number of bands
    print('Comparing CSLC number of bands ...')
    if not nbands_ref == nbands_sec:
        print(f'ERROR Number of bands in reference CSLC {nbands_ref} differs'
              f'from number of bands {nbands_sec} in secondary CSLC')
        return

    print('Comparing geo transform arrays ...')
    if not np.array_equal(geotransform_ref, geotransform_sec):
        print(f'ERROR Reference geo transform array {dataset_ref} differs'
              f'from secondary CSLC geo transform array {dataset_sec}')
        return

    # Compare amplitude of reference and generated CSLC products
    slc_ref = dataset_ref.GetRasterBand(1).ReadAsArray()
    slc_sec = dataset_sec.GetRasterBand(1).ReadAsArray()

    diff_real = slc_ref.real - slc_sec.real
    diff_imag = slc_ref.imag - slc_sec.image
    length, width = diff_real.shape

    print('Check mean real part difference between CSLC products is < 1.0e-5')
    assert np.allclose(diff_real, np.zeros([length, width]),
                       atol=0.0, rtol=1.0e-5, equal_nan=True)
    print('Check mean imaginary part difference between CSLC products is < 1.0e-5')
    assert np.allclose(diff_imag, np.zeros([length, width]),
                       atol=0.0, rtol=1.0e-5, equal_nan=True)

    return


def compare_cslc_metadata(file_ref, file_sec):
    '''
    Compare reference and generated CSLC metadata
    '''

    # Check if metadata files exists
    if not os.path.exists(file_ref):
        print(f'ERROR reference CSLC metadata not found: {file_ref}')
        return

    if not os.path.exists(file_sec):
        print(f'ERROR CSLC metadata not found: {file_sec}')
        return

    # Load metadata
    with open(file_ref, 'r') as f:
        metadata_ref = json.load(f)

    with open(file_sec, 'r') as f:
        metadata_sec = json.load(f)

    # Intersect metadata keys
    set_ref_minus_sec = metadata_ref.keys() - metadata_sec.keys()
    set_sec_minus_ref = metadata_sec.keys() - metadata_ref.keys()

    err_str = "Metadata keys do not match.\n"
    if set_ref_minus_sec:
        err_str += f'Reference CSLC metadata extra entries: {set_ref_minus_sec}'
    if set_sec_minus_ref:
        err_str += f'Secondary CSLC metadata extra entries: {set_sec_minus_ref}'
    # Check if metadata key differ
    assert (not set_ref_minus_sec or not set_sec_minus_ref, err_str)

    # Check remaining metadatakeys
    for k_ref, v_ref in metadata_ref.items():
        if metadata_sec[k_ref] != v_ref:
            print(f'ERROR the content of metadata key {k_ref} from'
                  f'reference CSLC metadata has a value {v_ref} whereas the same'
                  f'key in the secondary CSLC metadata has value {metadata_sec[k_ref]}')


if __name__ == '__main__':
    cmd = cmd_line_parser()

    # Check CSLC products
    compare_cslc_products(cmd.ref_product, cmd.sec_product)
    print('All CSLC product checks have passed')

    # Check CSLC metadata
    compare_cslc_metadata(cmd.ref_metadata, cmd.sec_metadata)
    print('All CSLC metadata checks have passed')

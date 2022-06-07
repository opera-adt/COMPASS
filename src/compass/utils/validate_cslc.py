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


def compare_cslc_products(file_1, file_2):
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
    if not os.path.exists(file_1):
        print(f'ERROR reference CSLC product not found: {file_1}')
        return

    if not os.path.exists(file_2):
        print(f'ERROR secondary CSLC product not found: {file_2}')
        return

    # Extract some info from reference/secondary CSLC products
    dataset_1 = gdal.Open(file_1, gdal.GA_ReadOnly)
    geotransform_1 = dataset_1.GetGeoTransform()
    nbands_1 = dataset_1.RasterCount

    dataset_2 = gdal.Open(file_2, gdal.GA_ReadOnly)
    geotransform_2 = dataset_2.GetGeoTransform()
    nbands_2 = dataset_2.RasterCount

    # Compare number of bands
    print('Comparing CSLC number of bands ...')
    if not nbands_1 == nbands_2:
        print(f'ERROR Number of bands in reference CSLC {nbands_1} differs'
              f'from number of bands {nbands_2} in secondary CSLC')
        return

    print('Comparing geo transform arrays ...')
    if not np.array_equal(geotransform_1, geotransform_2):
        print(f'ERROR Reference geo transform array {geotransform_1} differs'
              f'from secondary CSLC geo transform array {geotransform_2}')
        return

    # Compare amplitude of reference and generated CSLC products
    slc_1 = dataset_1.GetRasterBand(1).ReadAsArray()
    slc_2 = dataset_2.GetRasterBand(1).ReadAsArray()

    diff_amp = np.abs(slc_1) - np.abs(slc_2)
    diff_pha = np.angle(slc_1 * np.conj(slc_2))

    print('Check max amplitude difference between CSLC products is < 1.0e-12')
    assert np.nanmax(diff_amp) < 1.0e-12
    print('Check max phase difference between CSLC products is < 1.0e-12')
    assert np.nanmax(diff_pha) < 1.0e-12
    return


def compare_cslc_metadata(file_1, file_2):
    '''
    Compare reference and generated CSLC metadata
    '''

    # Check if metadata files exists
    if not os.path.exists(file_1):
        print(f'ERROR reference CSLC metadata not found: {file_1}')
        return

    if not os.path.exists(file_2):
        print(f'ERROR CSLC metadata not found: {file_2}')
        return

    # Load metadata
    with open(file_1, 'r') as f:
        metadata_1 = json.load(f)

    with open(file_2, 'r') as f:
        metadata_2 = json.load(f)

    print('Compare number of metadata keys')
    if not len(metadata_1.keys()) == len(metadata_2.keys()):
        print('ERROR different number of metadata keys')
        return

    # Intersect metadata keys
    set_1_m_2 = set(metadata_1.keys()) - set(metadata_2.keys())
    if len(set_1_m_2) > 0:
        print(f'Reference CSLC metadata has extra entries with keys:'
              f'{", ".join(set_1_m_2)}.')
        return
    set_2_m_1 = set(metadata_2.keys()) - set(metadata_1.keys())
    if len(set_2_m_1) > 0:
        print(f'Secondary CSLC metadata has extra entries with keys:'
              f'{", ".join(set_2_m_1)}.')

    # Check remaining metadatakeys
    for k1, v1 in metadata_1.items():
        if k1 not in metadata_2.keys():
            print(f'ERROR the metadata key {key1} in not present'
                  f'in the secondary CSLC metadata')
            return
        if metadata_2[k1] != v1:
            print(f'ERROR the content of metadata key {k1} from'
                  f'reference CSLC metadata has a value {v1} whereas the same'
                  f'key in the secondary CSLC metadata has value {metadata_2[k1]}')


if __name__ == '__main__':
    cmd = cmd_line_parser()

    # Check CSLC products
    compare_cslc_products(cmd.ref_product, cmd.sec_product)
    print('All CSLC product checks have passed')

    # Check CSLC metadata
    compare_cslc_metadata(cmd.ref_metadata, cmd.sec_metadata)
    print('All CSLC metadata checks have passed')

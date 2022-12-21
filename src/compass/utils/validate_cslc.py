import argparse
import os

import h5py
import numpy as np
from osgeo import gdal


def cmd_line_parser():
    """
    Command line parser
    """
    parser = argparse.ArgumentParser(description="""Validate
                                     reference and generated (secondary) S1 CSLC products""",
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
    raster_key = 'SLC'

    # Extract polarization with h5py
    with h5py.File(path_h5) as h:
        # convert keyview to list to get polarization key
        pol = list(h[raster_key].keys())[0]

    # Extract some info from reference/secondary CSLC products with GDAL
    h5_gdal_path = f'HDF5:{path_h5}://{raster_key}/{pol}'
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

    # Compare amplitude of reference and generated CSLC products
    print('Check mean real part difference between CSLC products is < 1.0e-5')
    assert np.allclose(slc_ref.real, slc_sec.real,
                       atol=0.0, rtol=1.0e-5, equal_nan=True)
    print('Check mean imaginary part difference between CSLC products is < 1.0e-5')
    assert np.allclose(slc_ref.imag, slc_sec.imag,
                       atol=0.0, rtol=1.0e-5, equal_nan=True)


def _get_metadata_keys(path_h5):
    """
    Extract metadata group/dataset names of a given HDF5

    Parameters
    ----------
    path_h5: str
        File path to CSLC HDF5 product

    Returns
    -------
    metadata_dict: str
        Dict holding metadata group and dataset names where:
        keys: metadata key names representing datasets or groups
        values: set of key names belonging to each metadata key names
    """
    metadata_dict = {}
    with h5py.File(path_h5, 'r') as h:
        metadata = h['metadata']
        # get metadata keys and iterate
        metadata_keys = set(metadata.keys())
        for metadata_key in metadata_keys:
            if not isinstance(metadata[metadata_key], h5py.Group):
                continue
            # save keys to current metadata key
            metadata_dict[metadata_key] = set(metadata[metadata_key].keys())

    return metadata_dict


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
    metadata_ref = _get_metadata_keys(file_ref)
    metadata_sec = _get_metadata_keys(file_sec)

    # Intersect metadata keys
    set_ref_minus_sec = metadata_ref.keys() - metadata_sec.keys()
    set_sec_minus_ref = metadata_sec.keys() - metadata_ref.keys()

    err_str = "Metadata keys do not match.\n"
    if set_ref_minus_sec:
        err_str += f'\nReference CSLC metadata extra entries: {set_ref_minus_sec}'
    if set_sec_minus_ref:
        err_str += f'\nSecondary CSLC metadata extra entries: {set_sec_minus_ref}'

    # Check if metadata key differ
    assert not set_ref_minus_sec or not set_sec_minus_ref, err_str

    # Check sub metadatakeys (after establishing top level metadata matches)
    for key in metadata_ref.keys():
        # Intersect metadata keys
        set_ref_minus_sec = metadata_ref[key] - metadata_sec[key]
        set_sec_minus_ref = metadata_sec[key] - metadata_ref[key]

        err_str = "Metadata keys do not match.\n"
        if set_ref_minus_sec:
            err_str += f'\nReference CSLC {key} metadata extra entries: {set_ref_minus_sec}'
        if set_sec_minus_ref:
            err_str += f'\nSecondary CSLC {key} metadata extra entries: {set_sec_minus_ref}'

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

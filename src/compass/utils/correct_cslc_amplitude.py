#! python

'''
Script to mosaic the input CSLC rasters
'''
import argparse

import numpy as np
from osgeo import gdal, osr

# Constants for dataset location in HDF5 file
PATH_CSLC_LAYER_IN_HDF = '/data'
PATH_LOCAL_INCIDENCE_ANGLE = '/data/local_incidence_angle'
PATH_NOISE_LAYER_IN_HDF = '/metadata/noise_information'


def _get_epsg(gdal_raster_path: str):
    '''
    Get the EPSG of the input raster
    '''
    ds_input = gdal.Open(gdal_raster_path, gdal.GA_ReadOnly)
    projection = ds_input.GetProjection()
    srs = osr.SpatialReference(wkt=projection)
    return int(srs.GetAuthorityCode(None))


def load_amplitude(ds_cslc_amp, ds_local_incidence_angle=None, ds_noise_lut=None):
    '''
    Load the amplitude from CSLC GDAL dataset.
    Apply noise correction and/or radiometric normalization when
    the data are provided.

    Parameters
    ----------
    ds_cslc_amp: osgeo.gdal.Dataset
        GDAL Raster dataset for CSLC amplitude
    ds_local_incidence_angle: osgeo.gdal.Dataset
        GDAL Raster dataset for local incidence angle
    ds_noise_lut: osgeo.gdal.Dataset
        GDAL Raster dataset noise LUT

    Returns
    -------
    arr_cslc: np.ndarray
        CSLC amplitude
    '''

    # Load CSLC amplutude into array
    arr_cslc = ds_cslc_amp.ReadAsArray()

    if ds_noise_lut is not None:
        print('Applying noise removal')
        # Apply noise correction

        arr_cslc = arr_cslc ** 2 - ds_noise_lut.ReadAsArray()
        arr_cslc[arr_cslc<0.0] = 0.0
        arr_cslc = np.sqrt(arr_cslc)

    if ds_local_incidence_angle is not None:
        print('Applying radiometric mormalization')
        local_incidence_angle_arr_rad = np.deg2rad(ds_local_incidence_angle.ReadAsArray())
        # Apply radiometric normalization using cotangent(local incidence angle)
        correction_factor = (np.sin(local_incidence_angle_arr_rad)
                             / np.cos(local_incidence_angle_arr_rad))
        arr_cslc *= correction_factor

    return arr_cslc


def get_cslc_gdal_dataset(cslc_path, cslc_static_path, pol,
                          noise_correction=True,
                          radiometric_normalization=True,
                          resampling_alg='BILINEAR'):
    '''
    Get the GDAL dataset for CSLC layer and resample &
    reprojecte noise LUT (when user opted in)

    Parameters
    ----------
    cslc_path: str
        path to the CSLC HDF5 file
    cslc_static_path: str
        path to the CSLC static layer HDF5 file
    pol: str
        Polarization
    noise_correction: bool
        Flag to turn on/off noise correction
    radiometric_normalization:bool
        Flag to turn on/ott radiometric normalization
    resampling_alg: str
        Resampling algorithm for gdal.Warp()

    Returns
    -------
    (ds_cslc,
     ds_incidence_angle,
     ds_noise_resampled): tuple
     Respectively, GDAL raster dataset for CSLC,
     GDAL raster dataset for local incidence angle, and
     GDAL raster dataset for noise LUT
     (resampled to the same geogrid as the other two)
    '''
    # Get the geotransform and dimensions


    prefix_netcdf = 'DERIVED_SUBDATASET:AMPLITUDE:NETCDF'

    path_cslc = f'{prefix_netcdf}:{cslc_path}:{PATH_CSLC_LAYER_IN_HDF}/{pol}'
    path_local_incidence = f'NETCDF:{cslc_static_path}:{PATH_LOCAL_INCIDENCE_ANGLE}'
    path_noise_lut = f'NETCDF:{cslc_path}:{PATH_NOISE_LAYER_IN_HDF}/thermal_noise_lut'

    ds_in = gdal.Open(path_cslc, gdal.GA_ReadOnly)

    ds_noise_lut = (gdal.Open(path_noise_lut, gdal.GA_ReadOnly) if
                        noise_correction else None)

    ds_local_incidence_angle = (gdal.Open(path_local_incidence, gdal.GA_ReadOnly) if
                        radiometric_normalization else None)

    # Prepare for the reprojection
    # Reproject the CSLC layer, and
    # Define the source and target spatial references
    gt_in = ds_in.GetGeoTransform()
    xsize = ds_in.RasterXSize
    ysize = ds_in.RasterYSize

    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(ds_in.GetProjectionRef())

    # Convert the corner points in `epsg_out`
    extent_reprojected=[gt_in[0],
                        gt_in[3] + gt_in[5] * ysize,
                        gt_in[0] + gt_in[1] * xsize,
                        gt_in[3]]

    # Put together the warp options
    warp_options = gdal.WarpOptions(format='MEM',
                                    resampleAlg=resampling_alg,
                                    xRes=gt_in[1],
                                    yRes=abs(gt_in[5]),
                                    outputBounds=extent_reprojected,
                                    dstSRS=src_srs)

    # Resample the noise LUT
    ds_noise_resampled = (gdal.Warp('', ds_noise_lut, options=warp_options) if
                          ds_noise_lut else None)

    return (ds_in,
            ds_local_incidence_angle,
            ds_noise_resampled)


def save_amplitude(amplitude_arr, geotransform_cslc, epsg_cslc,
                   amplitude_filename):
    '''
    Save mosaicked array into GDAL Raster

    Parameters
    ----------
    amplitude_arr: np.ndarray
        Mosaicked image as numpy array
    geotransform_cslc: tuple
        Geotransform parameter for mosaic
    epsg_cslc:
        EPSG for mosaic as projection parameter
    amplitude_filename:
        File name of the mosaic raster to save
    '''

    # Create the projection from the EPSG code
    srs_out = osr.SpatialReference()
    srs_out.ImportFromEPSG(epsg_cslc)
    projection_out = srs_out.ExportToWkt()

    length_mosaic, width_mosaic = amplitude_arr.shape
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(amplitude_filename,
                           width_mosaic, length_mosaic, 1, gdal.GDT_Float32,
                           options=['COMPRESS=LZW', 'BIGTIFF=YES'])

    ds_out.SetGeoTransform(geotransform_cslc)
    ds_out.SetProjection(projection_out)

    ds_out.GetRasterBand(1).WriteArray(amplitude_arr)

    ds_out = None


def get_parser():
    '''
    Get the parser for CLI
    '''
    parser = argparse.ArgumentParser(
        description='Comparison script with burst ID',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c',
                        dest='cslc_path',
                        type=str,
                        default='',
                        help='cslc file')

    parser.add_argument('-s',
                        dest='cslc_static_path',
                        type=str,
                        default=[],
                        help='CSLC static layer file')

    parser.add_argument('-o',
                        dest='out_path',
                        type=str,
                        default=None,
                        help='path to the output file')

    parser.add_argument('-p',
                        dest='pol',
                        type=str,
                        default='VV',
                        help='Polarization')

    parser.add_argument('--noise_correction_off',
                        dest='apply_noise_correction',
                        default=True,
                        action='store_false',
                        help='Turn off the noise correction')

    parser.add_argument('--radiometric_normalization_off',
                        dest='apply_radiometric_normalization',
                        default=True,
                        action='store_false',
                        help='Turn off the radiometric normalization')

    return parser


def main():
    '''
    Entrypoint of the fiunction
    '''
    parser = get_parser()
    args = parser.parse_args()

    cslc_layer, noise_lut, local_incidence_angle = \
        get_cslc_gdal_dataset(args.cslc_path,
                              args.cslc_static_path,
                              args.pol,
                              args.apply_noise_correction,
                              args.apply_radiometric_normalization)
    amplitude_arr = load_amplitude(cslc_layer, noise_lut, local_incidence_angle)

    geotransform_cslc = cslc_layer.GetGeoTransform()

    proj_cslc = cslc_layer.GetProjection()
    srs_cslc = osr.SpatialReference(wkt=proj_cslc)
    epsg_cslc = int(srs_cslc.GetAuthorityCode(None))

    save_amplitude(amplitude_arr, geotransform_cslc, epsg_cslc, args.out_path)


if __name__=='__main__':
    main()

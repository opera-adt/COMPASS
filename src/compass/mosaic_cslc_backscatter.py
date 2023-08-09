'''
Script to mosaic the input CSLC rasters
'''
import argparse
import os
from collections import Counter

import numpy as np
from osgeo import gdal, osr

# Constants for dataset location in HDF5 file
PATH_CSLC_LAYER_IN_HDF = '/data'
PATH_LOCAL_INCIDENCE_ANGLE = '/data/local_incidence_angle'
PATH_NOISE_LAYER_IN_HDF = '/metadata/noise_information'


def get_parser():
    '''
    Get the parser for CLI
    '''
    parser = argparse.ArgumentParser(
        description='Comparison script with burst ID',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c',
                        dest='cslc_list_file',
                        type=str,
                        default='',
                        help='cslc list file')

    parser.add_argument('-s',
                        dest='static_list_file',
                        type=str,
                        default=[],
                        help='static list file')

    parser.add_argument('-o',
                        dest='mosaic_out_path',
                        type=str,
                        default=None,
                        help='path to the burst DB')

    parser.add_argument('-cd',
                        dest='cslc_directory',
                        type=str,
                        default=None,
                        help='path to the cslc directory')

    parser.add_argument('-sd',
                        dest='cslc_static_directory',
                        type=str,
                        default=None,
                        help='path to the cslc static layer directory')

    return parser


def get_most_frequent_epsg(gdal_raster_list: list):
    '''
    Get the most frequent EPSG among the rasters in 'gdal_raster_list'

    Parameters
    ----------
    gdal_raster_list: list
        List of strings for GDAL raster dataset

    Returns
    -------
    most_common_epsg: int
        Most common EPSG code among the input parameters

    '''
    # Get a list of all EPSG codes
    epsg_list = [_get_epsg(raster) for raster in gdal_raster_list]

    # Count the occurrence of each EPSG code
    counter = Counter(epsg_list)

    # Return the most common EPSG code
    most_common_epsg = counter.most_common(1)[0][0]

    return most_common_epsg


def _get_epsg(gdal_raster_path: str):
    '''
    Get the EPSG of the input raster
    '''
    ds_input = gdal.Open(gdal_raster_path, gdal.GA_ReadOnly)
    projection = ds_input.GetProjection()
    srs = osr.SpatialReference(wkt=projection)
    return int(srs.GetAuthorityCode(None))


def get_cslc_gdal_dataset(cslc_path, cslc_static_path, pol, epsg_out, dx, dy, snap=0,
                      noise_correction=True,
                      radiometric_normalization=True,
                      resampling_alg='BILINEAR'):
    '''
    Get the GDAL dataset for CSLC layer and resampled & reprojected noise LUT (when user opted in)

    '''
    # Get the geotransform and dimensions
    prefix_netcdf = 'DERIVED_SUBDATASET:AMPLITUDE:NETCDF'

    path_cslc = f'{prefix_netcdf}:{cslc_path}:{PATH_CSLC_LAYER_IN_HDF}/{pol}'
    path_local_incidence = f'NETCDF:{cslc_static_path}:{PATH_LOCAL_INCIDENCE_ANGLE}'
    path_noise_lut = f'NETCDF:{cslc_path}:{PATH_NOISE_LAYER_IN_HDF}/thermal_noise_lut'

    ds_in = gdal.Open(path_cslc, gdal.GA_ReadOnly)
    epsg_cslc = _get_epsg(path_cslc)

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
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(epsg_out)

    # Define a coordinate transformation
    transform = osr.CoordinateTransformation(src_srs, dst_srs)

    # Define the corner points
    corners = [(0, 0), (0, ysize), (xsize, ysize), (xsize, 0)]

    # Convert the corner points in `epsg_out`
    transformed_corners = []

    # Compute the extet of the resampled raster
    if epsg_cslc == epsg_out:
        extent_reprojected=[gt_in[0],
                            gt_in[3] + gt_in[5] * ysize,
                            gt_in[0] + gt_in[1] * xsize,
                            gt_in[3]]
    else:
        for corner in corners:
            x_from, y_from = (gt_in[0] + corner[0] * gt_in[1],
                              gt_in[3] + corner[1] * gt_in[5])
            x_transformed, y_transformed, _ = transform.TransformPoint(x_from, y_from)
            transformed_corners.append([x_transformed, y_transformed])
        transformed_corners = np.array(transformed_corners)

        if snap != 0:
            transformed_corners = np.round(transformed_corners / snap) * snap

        # Extract the extent of the reprojected raster
        extent_reprojected = [np.min(transformed_corners[:, 0]),
                              np.min(transformed_corners[:, 1]),
                              np.max(transformed_corners[:, 0]),
                              np.max(transformed_corners[:, 1])]

    # Put together the warp options
    warp_options = gdal.WarpOptions(format='MEM',
                                    resampleAlg=resampling_alg,
                                    xRes=dx,
                                    yRes=abs(dy),
                                    outputBounds=extent_reprojected,
                                    dstSRS=dst_srs)
    # Resample the noise LUT
    ds_noise_resampled = (gdal.Warp('', ds_noise_lut, options=warp_options) if
                          ds_noise_lut else None)

    if epsg_cslc == epsg_out:
        # Do not perform reprojection when EPSG of the burst is the same as that of the mosaic
        # return the GDAL Dataset objects as they are, except for noise LUT
        return (ds_in, ds_local_incidence_angle, ds_noise_resampled)

    ds_incidence_angle_reprojected = (gdal.Warp('', ds_local_incidence_angle, options=warp_options) if ds_local_incidence_angle else None)
    ds_cslc_reprojected = gdal.Warp('', ds_in, options=warp_options)
    

    # Return the warped dataset
    return (ds_cslc_reprojected, ds_incidence_angle_reprojected, ds_noise_resampled)


def load_amplitude(ds_cslc_amp, ds_local_incidence_angle=None, ds_noise_lut=None):
    '''
    Apply noise correction using the geocoded noist LUT in the product
    '''
    # Load CSLC amplutude into array
    arr_cslc = ds_cslc_amp.ReadAsArray()

    if ds_noise_lut is not None:
        print('Applying noise removal')
        # Resample noise LUT to the geogrid of the CSLC amplitude
        #geotransform_cslc = ds_cslc_amp.GetGeoTransform()
        #width_cslc = ds_cslc_amp.RasterXSize
        #height_cslc = ds_cslc_amp.RasterYSize

        #bbox_cslc=[geotransform_cslc[0],
        #        geotransform_cslc[3] + geotransform_cslc[5] * height_cslc,
        #        geotransform_cslc[0] + geotransform_cslc[1] * width_cslc,
        #        geotransform_cslc[3]]

        #ds_resampled_lut = gdal.Warp(
        #    destNameOrDestDS='',
        #    srcDSOrSrcDSTab=ds_noise_lut,
        #    format='MEM',
        #    outputBounds=bbox_cslc,
        #    width=width_cslc,
        #    height=height_cslc)

        # Apply noise correction

        arr_cslc = arr_cslc ** 2 - ds_noise_lut.ReadAsArray()
        arr_cslc[arr_cslc<0.0] = 0.0
        arr_cslc = np.sqrt(arr_cslc)

    if ds_local_incidence_angle is not None:
        print('Applying radiometric mormalization')
        # Apply radiometric normalization using local incidence angle
        correction_factor = np.cos(np.deg2rad(ds_local_incidence_angle.ReadAsArray()))
        arr_cslc /= correction_factor

    return arr_cslc


def find_upperleft_pixel_index(geotransform_burst, geotransform_mosaic):
    '''
    Find the upperleft corner of the burst image in mosaic image, in pixels

    Parameters
    ----------
    geotransform_burst: tuple
        Geotransform parameter for burst
    geotransform_mosaic: tuple
        Geotransform parameter for mosaic

    Returns
    -------
    upperleft_y_px, upperleft_x_px: int
        relative location of upper-left corner of the burst in mosaic
    '''
    xmin_burst = geotransform_burst[0]
    xmin_mosaic = geotransform_mosaic[0]

    ymax_burst = geotransform_burst[3]
    ymax_mosaic = geotransform_mosaic[3]

    upperleft_x_px = int((xmin_burst - xmin_mosaic) / geotransform_mosaic[1])
    upperleft_y_px = int((ymax_burst - ymax_mosaic) / geotransform_mosaic[5])

    return (upperleft_y_px, upperleft_x_px)


def mosaic_list_order_mode(cslc_raster_list, local_incidence_angle_list, noise_lut_raster_list,
                           geotransform_mosaic, shape_mosaic):
    '''
    Perform the mosaicking in "list order mode"
    i.e. The pixels from the earlier raster gets replaced by the following rasters
    '''

    array_mosaic = np.zeros(shape_mosaic, dtype=np.float32) / 0

    for i_raster, cslc_raster in enumerate(cslc_raster_list):
        print(f'Processing: {i_raster + 1} of {len(cslc_raster_list)}')
        noise_lut_raster = noise_lut_raster_list[i_raster]
        local_incidence_angle_raster = local_incidence_angle_list[i_raster]
        uly, ulx = find_upperleft_pixel_index(cslc_raster.GetGeoTransform(), geotransform_mosaic)
        amplitude_burst = load_amplitude(cslc_raster, local_incidence_angle_raster, noise_lut_raster)
        length_burst, width_burst = amplitude_burst.shape

        subset_array_mosaic = array_mosaic[uly:uly+length_burst, ulx:ulx+width_burst]
        mask_valid = ~np.isnan(amplitude_burst)
        subset_array_mosaic[mask_valid] = amplitude_burst[mask_valid]

    return array_mosaic


def mosaic_nearest_centroid_mode(cslc_raster_list, noise_lut_raster_list,
                                 geotransform_mosaic, shape_mosaic):
    '''
    Perform the mosaicking in "nearest centroid" mode

    '''
    array_mosaic = np.array(shape_mosaic, dtype=np.float32)

    
    return array_mosaic


def compute_mosaic_geotransform_dimension(ds_burst_list):
    """
    Compute the GeoTransform vector that covers all rasters in `raster_list`
    """

    # Initialize list to store extents
    if len(ds_burst_list) == 0:
        raise RuntimeError('Empty ds_burst_list was provided')

    extents = []
    for ds_burst in ds_burst_list:
        gt_burst = ds_burst.GetGeoTransform()
        # Compute extent from GeoTransform
        xmin = gt_burst[0]
        xmax = gt_burst[0] + gt_burst[1] * ds_burst.RasterXSize
        ymin = gt_burst[3] + gt_burst[5] * ds_burst.RasterYSize
        ymax = gt_burst[3]
        extents.append([xmin, xmax, ymin, ymax])

    extents = np.array(extents)

    # Find common extent
    xmin = np.min(extents[:, 0])
    xmax = np.max(extents[:, 1])
    ymin = np.min(extents[:, 2])
    ymax = np.max(extents[:, 3])

    # Create and return GeoTransform
    gt_burst = ds_burst_list[0].GetGeoTransform()
    gt_mosaic = (xmin, gt_burst[1], gt_burst[2], ymax, gt_burst[4], gt_burst[5])

    width_mosaic = int((xmax - xmin)/ gt_burst[1] + 0.5)
    height_mosaic = int((ymin - ymax)/ gt_burst[5] + 0.5)
    return gt_mosaic, (width_mosaic, height_mosaic)


def save_mosaicked_array(mosaic_arr, geotransform_mosaic, epsg_mosaic, mosaic_filename):
    '''
    Save mosaicked array into GDAL Raster

    Parameters
    ----------
    mosaic_arr: np.ndarray
        Mosaicked image as numpy array
    geotransform_mosaic: tuple
        Geotransform parameter for mosaic
    epsg_mosaic:
        EPSG for mosaic as projection parameter
    mosaic_filename:
        File name of the mosaic raster to save
    '''

    # Create the projection from the EPSG code
    srs_mosaic = osr.SpatialReference()
    srs_mosaic.ImportFromEPSG(epsg_mosaic)
    projection_mosaic = srs_mosaic.ExportToWkt()

    length_mosaic, width_mosaic = mosaic_arr.shape
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(mosaic_filename,
                           width_mosaic, length_mosaic, 1, gdal.GDT_Float32,
                           options=['COMPRESS=LZW', 'BIGTIFF=YES'])

    ds_out.SetGeoTransform(geotransform_mosaic)
    ds_out.SetProjection(projection_mosaic)

    ds_out.GetRasterBand(1).WriteArray(mosaic_arr)

    ds_out = None


def run(cslc_path_list, cslc_static_path_list, pol, mosaic_path,
        dx_mosaic=None, dy_mosaic=None, snap_meters=30,
        epsg_mosaic=None, mosaic_mode='list_order',
        apply_noise_correcion=True, apply_radiometric_normalization=True,
        resampling_alg='bicubic'):
    '''
    Workflow of the CSLC backscatter mosaic

    Parameters
    ----------
        cslc_path_list
        pol: str
            Polarization to mosaic
        mosaic_path:
            Path to the output mosaic
        dx_mosaic: float, Optional
            Spacing of the mosaic in x direction
        dy_mosaic:  float, Optional
            Spacing of the mosaic in y direction
        snap_meters:
            Snapping in meters
        epsg_mosaic: int, optional
            EPSG of the output mosaic.
            If None, then the EPSG will be determined based on the input bursts
        mosaic_mode:
            Mosaic mode
        apply_noise_correcion: bool
            Flag whether or not to apply thermal noise correction
        resampling_alg: str
            Resampling algorithm, if necessary
    '''

    # Determine the EPSG of the mosaic when it is not provided
    if epsg_mosaic is None:
        print('EPSG was not provided. Finding most common projection among the input rasters.')
        epsg_mosaic = get_most_frequent_epsg(
            [f'NETCDF:{cslc_path}:{PATH_CSLC_LAYER_IN_HDF}/{pol}' for cslc_path in cslc_path_list])
        print(f'EPSG of the mosaic will be: {epsg_mosaic}')

    if dx_mosaic is None or dy_mosaic is None:
        ds_1st_cslc = gdal.Open(f'NETCDF:{cslc_path_list[0]}:{PATH_CSLC_LAYER_IN_HDF}/{pol}')
        gt_1st_cslc = ds_1st_cslc.GetGeoTransform()
        dx_mosaic = gt_1st_cslc[1] if dx_mosaic is None else dx_mosaic
        dy_mosaic = gt_1st_cslc[5] if dy_mosaic is None else dy_mosaic

    # Get the list of gdal raster dataset
    num_raster_in = len(cslc_path_list)
    cslc_amp_ds_list = [None] * num_raster_in
    noise_lut_ds_list = [None] * num_raster_in
    local_incidence_angle_ds_list = [None] * num_raster_in

    for i_cslc, cslc_path in enumerate(cslc_path_list):
        cslc_static_path = cslc_static_path_list[i_cslc]
        print('Loading CSLC layer in amplutude, and nouise LUTs: '
              f'{i_cslc + 1} / {len(cslc_path_list)}',
              end='\r')
        datasets_cslc = get_cslc_gdal_dataset(cslc_path, cslc_static_path, pol, epsg_mosaic,
                                              dx_mosaic, dy_mosaic, snap_meters,
                                              apply_noise_correcion, apply_radiometric_normalization, 
                                              resampling_alg)
        cslc_amp_ds_list[i_cslc] = datasets_cslc[0]
        local_incidence_angle_ds_list[i_cslc] = datasets_cslc[1]
        noise_lut_ds_list[i_cslc] = datasets_cslc[2]

    print(' ')
    # Get the size of mosaic
    gt_mosaic, (width_mosaic, height_mosaic) = \
        compute_mosaic_geotransform_dimension(cslc_amp_ds_list)

    # Compute the array for the mosaicked image
    mosaic_functions = {
        'list_order': mosaic_list_order_mode,
        'nearest': mosaic_nearest_centroid_mode
    }

    if mosaic_mode not in mosaic_functions:
        raise NotImplementedError(f'Mosaicking was not implemented for mode: {mosaic_mode}')

    mosaic_arr = mosaic_functions[mosaic_mode](
        cslc_amp_ds_list, local_incidence_angle_ds_list, noise_lut_ds_list, gt_mosaic,
        (height_mosaic, width_mosaic))

    # Write out the mosaicked image
    print(f'Writing mosaic to: {mosaic_path}')
    save_mosaicked_array(mosaic_arr, gt_mosaic, epsg_mosaic, mosaic_path)


def main():
    '''
    Entrypoint of the fiunction
    '''
    parser = get_parser()
    args = parser.parse_args()

    run(list_burst, list_burst_static, 'VV', output_mosaic_path)






if __name__=='__main__':
    main()
    

'''
Script to mosaic the input CSLC rasters
'''
import os

import numpy as np
from osgeo import gdal, osr
from collections import Counter
from itertools import repeat

PATH_CSLC_LAYER_IN_HDF = '/data'
PATH_NOISE_LAYER_IN_HDF = '/metadata/noise_information'

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


def get_cslc_gdal_dataset(cslc_path, pol, epsg_out, dx, dy, snap=0,
                      noise_correction=True,
                      resampling_alg='bicubic'):
    '''
    Get the GDAL dataset for CSLC layer and resampled & reprojected noise LUT (when user opted in)

    '''
    # Get the geotransform and dimensions
    prefix_netcdf = f'DERIVED_SUBDATASET:AMPLITUDE:NETCDF'

    path_cslc = f'{prefix_netcdf}:{cslc_path}:{PATH_CSLC_LAYER_IN_HDF}/{pol}'
    path_noise_lut = f'NETCDF:{cslc_path}:{PATH_NOISE_LAYER_IN_HDF}/thermal_noise_lut'

    ds_in = gdal.Open(path_cslc, gdal.GA_ReadOnly)
    epsg_cslc = _get_epsg(path_cslc)

    ds_noise_lut = (gdal.Open(path_noise_lut, gdal.GA_ReadOnly) if
                        noise_correction else None)

    if epsg_cslc == epsg_out:
        # Do not perform reprojection,
        # and return the Dataset object diretly from the original
        ds_cslc_layer = ds_in
        return (ds_cslc_layer, ds_noise_lut)

    # Reproject the CSLC layer and
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
    for corner in corners:
        x, y = (gt_in[0] + corner[0] * gt_in[1],
                gt_in[3] + corner[1] * gt_in[5])
        x_transformed, y_transformed, _ = transform.TransformPoint(x, y)
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

    ds_cslc_reprojected = gdal.Warp('', ds_in, options=warp_options)
    ds_noise_reprojected = (gdal.Warp('', ds_noise_lut, options=warp_options) if
                            ds_noise_lut else None)

    # Return the warped dataset
    return (ds_cslc_reprojected, ds_noise_reprojected)


def apply_noise_correction():
    '''
    Apply noise correction using the geocoded noist LUT in the product
    '''
    pass


def mosaic_list_order_mode(gdal_raster_list, mosaic_path):
    pass


def mosaic_nearest_centroid_mode(gdal_raster_list, mosaic_path):
    pass


def compute_mosaic_geotransform_dimension(ds_list):
    """
    Compute the GeoTransform vector that covers all rasters in `raster_list`
    """

    # Initialize list to store extents
    if len(ds_list) == 0:
        raise RuntimeError('Empty ds_list was provided')

    extents = []
    for ds in ds_list:
        gt = ds.GetGeoTransform()
        # Compute extent from GeoTransform
        xmin = gt[0]
        xmax = gt[0] + gt[1] * ds.RasterXSize
        ymin = gt[3] + gt[5] * ds.RasterYSize
        ymax = gt[3]
        extents.append([xmin, xmax, ymin, ymax])

    extents = np.array(extents)

    # Find common extent
    xmin = np.min(extents[:, 0])
    xmax = np.max(extents[:, 1])
    ymin = np.min(extents[:, 2])
    ymax = np.max(extents[:, 3])

    # Create and return GeoTransform
    common_gt = (xmin, gt[1], gt[2], ymax, gt[4], gt[5])

    width_mosaic = int((xmax - xmin)/ gt[1] + 0.5)
    height_mosaic = int((ymin - ymax)/ gt[5] + 0.5)
    return common_gt, (width_mosaic, height_mosaic)



def run(cslc_path_list, pol, mosaic_path,
        dx_mosaic=None, dy_mosaic=None, snap_meters=30,
        epsg_mosaic=None, mode='list_order',
        apply_noise_correcion=True, resampling_alg='bicubic'):
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
        mode:
            Mosaic mode
        apply_noise_correcion: bool
            Flag whether or not to apply thermal noise correction
        resampling_alg: str
            Resampling algorithm, if necessary

    '''

    # Determine the EPSG of the mosaic when it is not provided
    if epsg_mosaic is None:
        epsg_mosaic = get_most_frequent_epsg(
            [f'NETCDF:{cslc_path}:{PATH_CSLC_LAYER_IN_HDF}/{pol}' for cslc_path in cslc_path_list])

    if dx_mosaic is None or dy_mosaic is None:
        ds_1st_cslc = gdal.Open(f'NETCDF:{cslc_path_list[0]}:{PATH_CSLC_LAYER_IN_HDF}/{pol}')
        gt_1st_cslc = ds_1st_cslc.GetGeoTransform()
        dx_mosaic = gt_1st_cslc[1] if dx_mosaic is None else dx_mosaic
        dy_mosaic = gt_1st_cslc[5] if dy_mosaic is None else dy_mosaic

    # Get the list of gdal raster dataset
    num_raster_in = len(cslc_path_list)
    cslc_amp_ds_list = [None] * num_raster_in
    noise_lut_ds_list = [None] * num_raster_in
    
    for i_cslc, cslc_path in enumerate(cslc_path_list):
        print(f'Loading CSLC layer in amplutude, and nouise LUTs: {i_cslc + 1} / {len(cslc_path_list)}', end='\r')
        datasets_cslc = get_cslc_gdal_dataset(cslc_path, pol, epsg_mosaic,
                                              dx_mosaic, dy_mosaic, snap_meters,
                                              apply_noise_correcion, resampling_alg)
        cslc_amp_ds_list[i_cslc] = datasets_cslc[0]
        noise_lut_ds_list[i_cslc] = datasets_cslc[1]

    print(' ')
    # Get the size of mosaic
    gt_mosaic, (width_mosaic, height_mosaic) = compute_mosaic_geotransform_dimension(cslc_amp_ds_list)

    print('sdfsadf')

    # Compute the array for the mosaicked image



    # Write out the mosaicked image


if __name__=='__main__':
    import glob
    HOMEDIR = os.getenv('HOME')
    burst_cslc_dir = os.path.join(HOMEDIR, 'Documents/OPERA_SCRATCH/CSLC/MOSAIC_TEST_SITE/output_s1_cslc','**/**/t*.h5')
    output_mosaic_path = os.path.join(HOMEDIR, 'Documents/OPERA_SCRATCH/CSLC/MOSAIC_TEST_SITE/output_s1_cslc/mosaic.tif')
    list_burst = glob.glob(burst_cslc_dir)
    run(list_burst, 'VV', output_mosaic_path)

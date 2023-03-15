import h5py
import isce3
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, osr
from PIL import Image
import s1reader
from shapely.geometry import Point
from skimage.transform import resize

from compass.utils.geo_runconfig import GeoRunConfig
from utils.calc import stats
from utils.h5_helpers import (Meta, add_dataset_and_attrs,
                              get_cslc_geo_transform)
from utils.raster_polygon import get_boundary_polygon

def pixel_validity_check(path_h5, bursts):
    root_path = '/science/SENTINEL1/CSLC'

    # init source and destination paths in hdf5
    src_paths = []
    dst_paths = []

    # add CSLC requiring stats
    grid_path = f'{root_path}/grids'
    for pol in [b.polarization for b in bursts]:
        src_paths.append(f'{grid_path}/{pol}')
        dst_paths.append(f'grid/{pol}')

    # compute stats and write to hdf5
    stat_names = ['number_invalid', 'number_in_bound']
    with h5py.File(path_h5, 'a', swmr=True) as h5_obj:
        qa_group = h5_obj.require_group(f'{root_path}/quality_assurance')
        for src_path, dst_path in zip(src_paths, dst_paths):
            # get data path and compute stats according to dtype
            ds_path = f'HDF5:%FILE_PATH%:/{src_path}'
            boundary = get_boundary_polygon(path_h5,
                                            dataset_path_template=ds_path)

            grid_group_path = '/science/SENTINEL1/CSLC/grids'
            grid_group = h5_obj[grid_group_path]
            x_coords_utm = grid_group['x_coordinates'][()]
            y_coords_utm = grid_group['y_coordinates'][()]

            arr = h5_obj[src_path][()]

            n_in_bound = 0
            n_invalid = 0
            # SLOOOOOOOOWWWWWWWWWWWWWWWWW!!!!!!!!!!!!!!!!!!!!
            for i_row, row in enumerate(arr):
                y_coord = y_coords_utm[i_row]
                for i_col, val in enumerate(row):
                    coord = Point(x_coords_utm[i_col], y_coord)
                    if boundary.contains(coord):
                        n_in_bound += 1
                        if isnan(val):
                            n_invalid += 1

            # write stats to HDF5
            vals = [n_invalid, n_in_bound]
            for ds_name, val in zip(stat_names, vals):
                desc = f'{ds_name} of {dst_path}'
                add_dataset_and_attrs(qa_group, Meta(ds_name, val, desc))


def scale_to_max_pixel_dimension(orig_shape, max_dim=2048):
    scaled_max = max([xy / max_dim for xy in orig_shape])
    scaled_shape = [int(np.ceil(xy / scaled_max)) for xy in orig_shape]
    return scaled_shape

def resize_to_browse(arr, max_dim=2048):
    browse_shape = scale_to_max_pixel_dimension(arr.shape, max_dim)
    arr_browse = np.zeros(browse_shape)
    for i in range(browse_shape[0]):
        i_start = i * scale_max_int
        i_s_ = np.s_[i_start:i_start + scale_max_int]
        for j in range(browse_shape[1]):
            j_start = j * scale_max_int
            j_s_ = np.s_[j_start:j_start + scale_max_int]
            arr_browse[i, j] = np.mean(arr[i_s_, j_s_])
    return arr_browse

def browse_image(path_h5, bursts, image_scale='linear', percent_lo=None,
                 percent_hi=None, gamma=1.0):
    root_path = '/science/SENTINEL1/CSLC'

    # determine how to represent magnitude complex imagery
    if image_scale == 'linear':
        complex_to_real = np.abs
        std_multiplier = 4
    elif image_scale == 'log':
        def log_abs(x):
            return np.log(np.abs(x))
        complex_to_real = log_abs
        std_multiplier = 2

    # init containers source and destination paths in hdf5
    src_paths = []

    # add CSLCs requiring stats
    grid_path = f'{root_path}/grids'

    with h5py.File(path_h5, 'r', swmr=True) as h5_obj:
        grid_group = h5_obj[grid_path]

        # extract epsg
        epsg = int(grid_group['projection'][()])

        for b in bursts:
            # get polarization to extract geocoded raster
            pol = b.polarization

            # compute browse shape
            full_shape = grid_group[pol].shape
            full_h, full_w = full_shape
            browse_h, browse_w = scale_to_max_pixel_dimension(full_shape)

            # create in memory GDAL raster for GSLC as real value array
            driver = gdal.GetDriverByName('MEM')
            src_raster = driver.Create('mem_raster', full_w, full_h, 1,
                                       gdal.GDT_Float32)
            arr = complex_to_real(grid_group[pol][()])
            src_raster.WriteArray(arr)

            # set geotransform and epsg of source
            src_raster.SetGeoTransform(get_cslc_geo_transform(path_h5, pol))
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(epsg)
            src_raster.SetProjection(srs.ExportToWkt())

            # gdal wrap to right shape and EPSG
            ds_wgs84 = gdal.Warp('', src_raster, format='MEM',
                                 srcSRS=f'EPSG:{epsg}', dstSRS='EPSG:4326',
                                 width=browse_w, height=browse_h,
                                 resampleAlg = gdal.GRIORA_Bilinear,
                                 transformerOptions=['DST_METHOD=NO_GEOTRANSFORM'])
            '''
            transformerOptions=['SRC_METHOD=NO_GEOTRANSFORM',
                                'DST_METHOD=NO_GEOTRANSFORM'])
            '''
            arr = np.power(complex_to_real(ds_wgs84.ReadAsArray()), gamma)

            # prepare file output
            date = b.sensing_start.strftime('%Y-%m-%d')
            fname = f'{b.burst_id}_{image_scale}_{pol}_{date}.png'

            # get hi/lo values by percentile
            if percent_hi is None:
                percent_hi = 100
            vmax = np.nanpercentile(arr, percent_hi)

            if percent_lo is None:
                percent_lo = 0
            vmin = np.nanpercentile(arr, percent_lo)

            # clip based on hi/lo percentiles
            arr = np.clip(arr, a_min=vmin, a_max=vmax)

            # scale to 0-1 for gray scale
            arr = (arr - vmin) / (vmax - vmin)

            # scale to 1-255
            nan_mask = np.isnan(arr)
            arr = np.uint8(arr * (254)) + 1

            # set NaNs to 0
            arr[nan_mask] = 0

            # save to disk
            img = Image.fromarray(arr, mode='L')
            img.save(fname, transparency=0)


if __name__ == "__main__":
    import os
    import sys
    # TODO replace with argparse


    cfg = GeoRunConfig.load_from_yaml(sys.argv[1], workflow_name='s1_cslc_geo')

    h5_path = sys.argv[2]
    #stats(h5_path, cfg.bursts)
    #pixel_validity_check(h5_path, cfg.bursts)
    #browse_image(h5_path, cfg.bursts)
    browse_image(h5_path, cfg.bursts, 'linear', gamma=0.2)

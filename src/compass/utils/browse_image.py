'''
function to generate CSLC browse image and image manipulation helper functions
'''
import argparse

import h5py
import numpy as np
from PIL import Image
from osgeo import gdal

from compass.utils.geo_runconfig import GeoRunConfig
from compass.utils.h5_helpers import get_georaster_bounds, GRID_PATH


def _scale_to_max_pixel_dimension(orig_shape, max_dim_allowed=2048):
    '''
    Scale up or down length and width represented by a shape to a maximum
    dimension. The larger of length or width used to compute scaling ratio.

    Parameters
    ----------
    orig_shape: tuple[int]
        Shape (length, width) to be scaled
    max_dim_allowed: int
        Maximum dimension allowed for either length or width

    Returns
    -------
    _: list(int)
        Shape (length, width) scaled up or down from original shape
    '''
    # compute scaling ratio based on larger dimension
    scaling_ratio = max([xy / max_dim_allowed for xy in orig_shape])

    # scale original shape by scaling ratio
    scaled_shape = [int(np.ceil(xy / scaling_ratio)) for xy in orig_shape]
    return scaled_shape


def _clip_by_percentage(image, percent_low, percent_high):
    '''
    Clip image by low and high percentiles

    Parameters
    ----------
    image: np.ndarray
        Numpy array representing an image to be clipped
    percent_low: float
        Lower percentile of non-NaN pixels to be clipped
    percent_high: float
        Higher percentile of non-NaN pixels to be clipped

    Returns
    -------
    image: np.ndarray
        Clipped image
    vmin: float
        Minimum value of image determined by percent_low
    vmax_: float
        Maximum value of image determined by percent_high
    '''
    if percent_high <= percent_low:
        raise ValueError('upper percentile not > lower percentile')

    # get max/min values by percentile
    vmax = np.nanpercentile(image, percent_high)
    vmin = np.nanpercentile(image, percent_low)

    # clip if necessary
    if percent_low != 0.0 or percent_high != 100.0:
        image = np.clip(image, a_min=vmin, a_max=vmax)

    return image, vmin, vmax


def _normalize_apply_gamma(image, vmin, vmax, gamma=1.0):
    '''
    Normal and gamma correct an image array

    Parameters
    ----------
    image: np.ndarray
        Numpy array representing an image to be normalized and gamma corrected
    vmin: float
        Minimum value of image to be used to scale image to 0-1
    vmax_: float
        Maximum value of image to be used to scale image to 0-1
    gamma: float
        Exponent value used to gamma correct image

    Returns
    -------
    image: np.ndarray
        Normalized and gamma corrected image
    '''
    if vmax <= vmin:
        raise ValueError('maximum value not > minimum value')

    # scale to 0-1 for gray scale and then apply gamma correction
    image = (image - vmin) / (vmax - vmin)

    # scale to 0-1 for gray scale and then apply gamma correction
    if gamma != 1.0:
        image = np.power(image, gamma)

    return image


def _image_histogram_equalization(image, number_bins=256):
    '''
    Apply histogram equalization to an image array

    Parameters
    ----------
    image: np.ndarray
        Numpy array representing an image to be histogram equalized
    number_bins: int
        Number of histogram bins

    Returns
    -------
    image: np.ndarray
        Histogram equalized image

    Reference
    ---------
        http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    '''
    if number_bins <= 0:
        raise ValueError('number of histogram bins must be >= 1')

    mask = np.isnan(image)

    # get image histogram based on non-nan values
    image_histogram, bins = np.histogram(image[~mask].flatten(),
                                         number_bins, density=True)

    # cumulative distribution function
    cdf = image_histogram.cumsum()

    # normalize
    cdf = (number_bins-1) * cdf / cdf[-1]

    # use linear interpolation of cdf to find new pixel values
    image_eq = np.interp(image.flatten(), bins[:-1], cdf).reshape(image.shape)
    image_eq[mask] = np.nan

    return image


def _save_to_disk_as_greyscale(image, fname):
    '''
    Save image array as greyscale to file

    Parameters
    ----------
    image: np.ndarray
        Numpy array representing an image to be saved to png file
    fname: str
        File name of output browse image
    '''
    # scale to 1-255
    # 0 reserved for transparency
    nan_mask = np.isnan(image)
    image = np.uint8(image * (254)) + 1

    # set NaNs to 0
    image[nan_mask] = 0

    # save to disk in grayscale ('L')
    img = Image.fromarray(image, mode='L')
    img.save(fname, transparency=0)


def make_browse_image(filename, path_h5, bursts, complex_to_real='amplitude', percent_low=0.0,
                      percent_high=100.0, gamma=1.0, equalize=False):
    '''
    Make browse image(s) for geocoded CSLC raster(s)

    Parameters
    ----------
    filename: str
        File name of output browse image
    path_h5: str
        HDF5 file containing geocoded CSLC raster(s)
    bursts: list
        Burst(s) of geocoded CSLC raster(s)
    complex_to_real: str
        Method to convert complex float CSLC data to float data. Available
        methods: amplitude, intensity, logamplitude
    percent_low: float
        Lower percentile of non-NaN pixels to be clipped
    percent_high: float
        Higher percentile of non-NaN pixels to be clipped
    gamma: float
        Exponent value used to gamma correct image
    equalize: bool
        Enable/disable histogram equalization
    '''
    # determine how to transform complex imagery in gdal warp
    if complex_to_real not in ['amplitude', 'intensity', 'logamplitude']:
        raise ValueError(f'{complex_to_real} invalid complex to real transform')
    derived_ds_str = f'DERIVED_SUBDATASET:{complex_to_real.upper()}'

    # prepend transform to NETCDF path to grid
    derived_netcdf_to_grid = f'{derived_ds_str}:NETCDF:{path_h5}:/{GRID_PATH}'

    with h5py.File(path_h5, 'r', swmr=True) as h5_obj:
        grid_group = h5_obj[GRID_PATH]

        for b in bursts:
            # get polarization to extract geocoded raster
            pol = b.polarization

            # compute browse shape
            full_shape = grid_group[pol].shape
            browse_h, browse_w = _scale_to_max_pixel_dimension(full_shape)

            # create in memory GDAL raster for GSLC as real value array
            src_raster = f'{derived_netcdf_to_grid}/{pol}'

            min_x, max_x, min_y, max_y = get_georaster_bounds(path_h5, pol)

            # gdal warp to right geo extents, image shape and EPSG
            ds_wgs84 = gdal.Warp('', src_raster, format='MEM',
                                 dstSRS='EPSG:4326',
                                 width=browse_w, height=browse_h,
                                 resampleAlg = gdal.GRIORA_Bilinear,
                                 outputBounds=(min_x, min_y, max_x, max_y),
                                 )
            image = ds_wgs84.ReadAsArray()

            # get hi/lo values by percentile
            image, vmin, vmax = _clip_by_percentage(image, percent_low,
                                                    percent_high)

            if equalize:
                image = _image_histogram_equalization(image)

            # scale valid pixels to 1-255
            # set NaNs set to 0 to be transparent
            image = _normalize_apply_gamma(image, vmin, vmax, gamma)

            # save to disk
            _save_to_disk_as_greyscale(image, filename)


if __name__ == "__main__":
    '''Run geocode cslc workflow from command line'''
    # load arguments from command line
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('run_config_path', type=str, nargs='?',
                        default=None, help='Path to run config file')
    parser.add_argument('-o', '--out_fname', type=str, nargs='?',
                        default=None, help='Path to output png file')
    parser.add_argument('-c','--complex_to_real', type=str,
                        default='amplitude', choices=['amplitude', 'intensity',
                                                      'logamplitude'],
                        help='Method to convert complex data to real')
    parser.add_argument('-l', '--percent_low', type=float, nargs='?',
                        default=0.0, help='Lower %age of non-NaN pixels to be clipped')
    parser.add_argument('-u', '--percent_up', type=float, nargs='?',
                        default=100.0, help='Upper %age of non-NaN pixels to be clipped')
    parser.add_argument('-g', '--gamma', type=float, nargs='?',
                        default=1.0, help='Exponent value used for gamma correction')
    parser.add_argument('-e', '--equalize', type=bool, nargs='?',
                        default=False, help='Enable/disble histogram equalzation')
    args = parser.parse_args()

    # Get a runconfig dict from command line argumens
    cfg = GeoRunConfig.load_from_yaml(args.run_config_path,
                                      workflow_name='s1_cslc_geo')

    # unpack args to make browse image
    bursts = cfg.bursts
    burst = bursts[0]
    date_str = burst.sensing_start.strftime("%Y%m%d")
    burst_id_date_key = (str(burst.burst_id), date_str)
    out_paths = cfg.output_paths[burst_id_date_key]
    output_hdf5 = out_paths.hdf5_path

    # Run geocode burst workflow
    make_browse_image(args.out_fname, output_hdf5, bursts,
                      args.complex_to_real,  args.percent_low, args.percent_up,
                      args.gamma, args.equalize)

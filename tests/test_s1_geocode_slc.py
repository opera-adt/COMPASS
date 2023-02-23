import csv

import h5py
import isce3
import numpy as np

from compass import s1_geocode_slc
from compass.utils.geo_runconfig import GeoRunConfig


def test_geocode_slc_run(geocode_slc_params):
    '''
    Run s1_geocode_slc to ensure it does not crash

    Parameters
    ----------
    geocode_slc_params: SimpleNamespace
        SimpleNamespace containing geocode SLC unit test parameters
    '''
    # load yaml to cfg
    cfg = GeoRunConfig.load_from_yaml(geocode_slc_params.gslc_cfg_path,
                                      workflow_name='s1_cslc_geo')

    # pass cfg to s1_geocode_slc
    s1_geocode_slc.run(cfg)


def _get_nearest_index(arr, val):
    '''
    Find index of element in given array closest to given value

    Parameters
    ----------
    arr: np.ndarray
        1D array to be searched
    val: float
        Number to be searched for

    Returns
    -------
    _: int
        Index of element in arr where val is closest
    '''
    return np.abs(arr - val).argmin()


def _get_reflectors_bounding_slice(geocode_slc_params):
    '''
    Get latitude, longitude slice that contains all the corner reflectors in
    CSV list of corner reflectors

    Parameters
    ----------
    geocode_slc_params: SimpleNamespace
        SimpleNamespace containing geocode SLC unit test parameters
    '''
    # extract from HDF5
    with h5py.File(geocode_slc_params.output_hdf5, 'r') as h5_obj:
        grid_group = h5_obj[geocode_slc_params.grid_group_path]

        # create projection to covert from UTM to LLH
        epsg = int(grid_group['projection'][()])
        proj = isce3.core.UTM(epsg)

        x_coords_utm = grid_group['x_coordinates'][()]
        y_coords_utm = grid_group['y_coordinates'][()]

        lons = np.array([np.degrees(proj.inverse([x, y_coords_utm[0], 0])[0])
                         for x in x_coords_utm])

        lats = np.array([np.degrees(proj.inverse([x_coords_utm[0], y, 0])[1])
                         for y in y_coords_utm])

        # get array shape for later check of slice with margins applied
        height, width = h5_obj[geocode_slc_params.raster_path].shape

    # extract all lat/lon corner reflector coordinates
    corner_lats = []
    corner_lons = []
    with open(geocode_slc_params.corner_coord_csv_path, 'r') as csvfile:
        corner_reader = csv.DictReader(csvfile)
        for row in corner_reader:
            corner_lats.append(float(row['Latitude (deg)']))
            corner_lons.append(float(row['Longitude (deg)']))

    # find nearest index for min/max of lats/lons and apply margin
    # apply margin to bounding box and ensure raster bounds are not exceeded
    # application of margin y indices reversed due descending order lats vector
    margin = 50
    i_max_y = max(_get_nearest_index(lats, np.max(corner_lats)) - margin, 0)
    i_min_y = min(_get_nearest_index(lats, np.min(corner_lats)) + margin,
                  height - 1)
    i_max_x = min(_get_nearest_index(lons, np.max(corner_lons)) + margin,
                  width - 1)
    i_min_x = max(_get_nearest_index(lons, np.min(corner_lons)) - margin, 0)

    # return as slice
    # y indices reversed to account for descending order lats vector
    return np.s_[i_max_y:i_min_y, i_min_x:i_max_x]


def test_geocode_slc_validate(geocode_slc_params):
    '''
    Check for reflectors in geocoded output

    Parameters
    ----------
    geocode_slc_params: SimpleNamespace
        SimpleNamespace containing geocode SLC unit test parameters
    '''
    # get slice where corner reflectors should be
    s_ = _get_reflectors_bounding_slice(geocode_slc_params)

    # slice raster array
    with h5py.File(geocode_slc_params.output_hdf5, 'r') as h5_obj:
        arr = h5_obj[geocode_slc_params.raster_path][()][s_]

    # check for bright spots in sliced array
    corner_reflector_threshold = 3e3
    assert np.any(arr > corner_reflector_threshold)

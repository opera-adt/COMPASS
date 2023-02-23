import csv

import h5py
import isce3
import numpy as np

from compass import s1_geocode_slc
from compass.utils.geo_runconfig import GeoRunConfig


def test_geocode_slc_run(unit_test_paths):
    '''
    run s1_geocode_slc to ensure it does not crash
    '''
    # load yaml to cfg
    cfg = GeoRunConfig.load_from_yaml(unit_test_paths.gslc_cfg_path,
                                      workflow_name='s1_cslc_geo')

    # pass cfg to s1_geocode_slc
    s1_geocode_slc.run(cfg)

def get_nearest_index(arr, val):
    return np.abs(arr - val).argmin()

def get_reflectors_extents_slice(unit_test_paths, margin=50):
    '''
    get max and min lat, lon
    '''
    # extract from HDF5
    with h5py.File(unit_test_paths.output_hdf5, 'r') as h5_obj:
        grid_group = h5_obj[unit_test_paths.grid_group_path]

        # create projection to covert from UTM to LLH
        epsg = int(grid_group['projection'][()])
        proj = isce3.core.UTM(epsg)

        x_coords_utm = grid_group['x_coordinates'][()]
        y_coords_utm = grid_group['y_coordinates'][()]

        lons = np.array([np.degrees(proj.inverse([x, y_coords_utm[0], 0])[0])
                         for x in x_coords_utm])

        lats = np.array([np.degrees(proj.inverse([x_coords_utm[0], y, 0])[1])
                         for y in y_coords_utm])

    # extract all lat/lon corner reflector coordinates
    corner_lats = []
    corner_lons = []
    with open(unit_test_paths.corner_coord_csv_path, 'r') as csvfile:
        corner_reader = csv.DictReader(csvfile)
        for row in corner_reader:
            corner_lats.append(float(row['Latitude (deg)']))
            corner_lons.append(float(row['Longitude (deg)']))

    i_max_lat = get_nearest_index(lats, np.max(corner_lats))
    i_min_lat = get_nearest_index(lats, np.min(corner_lats))
    i_max_lon = get_nearest_index(lons, np.max(corner_lons))
    i_min_lon = get_nearest_index(lons, np.min(corner_lons))

    return np.s_[i_max_lat - margin:i_min_lat + margin,
                 i_min_lon - margin:i_max_lon + margin]

def test_geocode_slc_validate(unit_test_paths):
    '''
    check for reflectors in geocoded output
    '''
    s_ = get_reflectors_extents_slice(unit_test_paths)

    with h5py.File(unit_test_paths.output_hdf5, 'r') as h5_obj:
        src_path = f'{unit_test_paths.grid_group_path}/VV'
        arr = h5_obj[src_path][()][s_]
        print(arr.shape, s_)

    corner_reflector_threshold = 3e3
    assert np.any(arr > corner_reflector_threshold)

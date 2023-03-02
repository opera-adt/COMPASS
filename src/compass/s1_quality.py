import h5py
import isce3
import matplotlib.pyplot as plt
import numpy as np
import s1reader
from shapely.geometry import Point

from compass.utils.geo_runconfig import GeoRunConfig
from utils.h5_helpers import Meta, add_dataset_and_attrs
from utils.raster_polygon import get_boundary_polygon

def stats(path_h5, bursts):
    root_path = '/science/SENTINEL1/CSLC'

    # init containers source and destination paths in hdf5
    src_paths = []
    dst_paths = []

    # add CSLCs requiring stats
    grid_path = f'{root_path}/grids'
    for b in bursts:
        pol = b.polarization
        src_paths.append(f'{grid_path}/{pol}')
        dst_paths.append(f'grid/{pol}')

    # add corrections requiring stats
    corrections = ['bistatic_delay', 'geometry_steering_doppler',
                   'azimuth_fm_rate_mismatch']
    for correction in corrections:
        src_paths.append(f'{root_path}/corrections/{correction}')
        dst_paths.append(correction)

    # compute stats and write to hdf5
    stat_names = ['mean', 'min', 'max']
    with h5py.File(path_h5, 'a') as h5_obj:
        qa_group = h5_obj.require_group(f'{root_path}/quality_assurance')
        for src_path, dst_path in zip(src_paths, dst_paths):
            # get data path and compute stats according to dtype
            ds = h5_obj[src_path]
            if ds.dtype == 'complex64':
                stat_obj = isce3.math.StatsRealImagFloat32(ds[()])
                arr = ds[()]
                for typ, np_op in zip(['real', 'imag'], [np.real, np.imag]):
                    typ_arr = np_op(arr)

                # write stats to HDF5
                vals = []
                for cstat_member in [stat_obj.real, stat_obj.imag]:
                    vals.extend([cstat_member.mean, cstat_member.min,
                                 cstat_member.max])
                cstat_names = stat_names * 2
                for ds_name, val in zip(stat_names * 2, vals):
                    desc = f'{ds_name} of {dst_path}'
                    add_dataset_and_attrs(qa_group, Meta(ds_name, val, desc))
            else:
                stat_obj = isce3.math.StatsFloat32(ds[()].astype(np.float32))

                # write stats to HDF5
                vals = [stat_obj.mean, stat_obj.min, stat_obj.max]
                for ds_name, val in zip(stat_names, vals):
                    desc = f'{ds_name} of {dst_path}'
                    add_dataset_and_attrs(qa_group, Meta(ds_name, val, desc))


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

if __name__ == "__main__":
    import os
    import sys
    # TODO replace with argparse


    cfg = GeoRunConfig.load_from_yaml(sys.argv[1], workflow_name='s1_cslc_geo')

    h5_path = sys.argv[2]
    #stats(h5_path, cfg.bursts)
    pixel_validity_check(h5_path, cfg.bursts)

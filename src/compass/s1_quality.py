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
    stat_names = ['mean', 'min', 'max', 'std']
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
                                 cstat_member.max, cstat_member.sample_stddev])
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


def browse_image(path_h5, bursts):
    root_path = '/science/SENTINEL1/CSLC'

    # init containers source and destination paths in hdf5
    src_paths = []
    dst_paths = []

    # add CSLCs requiring stats
    grid_path = f'{root_path}/grids'

    with h5py.File(path_h5, 'r') as h5_obj:
        grid_group = h5_obj[grid_path]

        # extract axis coords and epsg
        x_coords_utm = grid_group['x_coordinates'][()]
        y_coords_utm = grid_group['y_coordinates'][()]
        epsg = grid_group['projection'][()]

        # create projection to convert axis to lat lon for plot extent
        proj = isce3.core.UTM(epsg)
        lons = np.array([np.degrees(proj.inverse([x, y_coords_utm[0], 0])[0])
                         for x in x_coords_utm])
        lats = np.array([np.degrees(proj.inverse([x_coords_utm[0], y, 0])[1])
                         for y in y_coords_utm])
        extent = [lons[0], lons[-1], lats[0], lats[-1]]

        qa_group = h5_obj[f'{root_path}/quality_assurance']

        for b in bursts:
            # get polarization to extract geocoded raster
            pol = b.polarization
            arr = np.abs(grid_group[pol][()])
            arr_nan_masked = np.ma.masked_array(arr, mask=np.isnan(arr))

            # prepare file output
            date = b.sensing_start.strftime('%Y-%m-%d')
            fname = f'{b.burst_id}_{pol}_{date}.png'

            # get stats needed to set max value of plot
            mean = np.mean(np.abs(arr_nan_masked))
            std = np.std(np.abs(arr_nan_masked))
            vmax = mean + 4 * std

            # plot and save to disk
            plt.close('all')
            plt.figure(figsize=(20,10))
            plt.imshow(np.abs(arr_nan_masked), vmax=vmax, extent=extent)
            plt.colorbar(orientation='horizontal')
            plt.xlabel('longitude (deg)')
            plt.ylabel('latitude (deg)')
            plt.title(f'CSLC {b.burst_id} {pol} {b.sensing_start}')
            plt.tight_layout()
            plt.savefig(fname, facecolor='white', edgecolor='none')


if __name__ == "__main__":
    import os
    import sys
    # TODO replace with argparse


    cfg = GeoRunConfig.load_from_yaml(sys.argv[1], workflow_name='s1_cslc_geo')

    h5_path = sys.argv[2]
    #stats(h5_path, cfg.bursts)
    #pixel_validity_check(h5_path, cfg.bursts)
    browse_image(h5_path, cfg.bursts)

#!/usr/bin/env python
import argparse
import datetime
import glob
import os
import time
from collections import defaultdict

import isce3
import journal
import numpy as np
import pandas as pd
import yaml
from s1reader.s1_orbit import get_orbit_file_from_dir, parse_safe_filename
from s1reader.s1_reader import load_bursts
from shapely import geometry

from compass.utils import helpers
from compass.utils.geo_grid import get_point_epsg


def create_parser():
    parser = argparse.ArgumentParser(
        description='S1-A/B geocoded CSLC stack processor.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Separate the required options from the optional ones
    # https://stackoverflow.com/a/41747010/
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('-s', '--slc-dir', required=True,
                          help='Directory containing the S1-A/B SLCs (zip files)')
    required.add_argument('-d', '--dem-file', required=True,
                          help='File path to a GDAL-readable DEM to use for processing.')
    optional.add_argument('-o', '--orbit-dir', default=None,
                          help='Directory with orbit files. If None, downloads orbit files')
    optional.add_argument('-w', '--working-dir', dest='work_dir', default='stack',
                          help='Directory to store intermediate and final results')
    optional.add_argument('-sd', '--start-date', help='Start date of the stack to process')
    optional.add_argument('-ed', '--end-date', help='End date of the stack to process')
    optional.add_argument('-b', '--burst-id', nargs='+', default=None,
                          help='List of burst IDs to process. If None, burst IDs '
                             'common to all dates are processed.')
    optional.add_argument('-exd', '--exclude-dates', nargs='+',
                          help='Date to be excluded from stack processing (format: YYYYMMDD)')
    optional.add_argument('-p', '--pol', dest='pol', nargs='+', default='co-pol',
                          help='Polarization to process: dual-pol, co-pol, cross-pol '
                               ' (default: co-pol).')
    optional.add_argument('-dx', '--x-spac', type=float, default=5,
                          help='Spacing in meters of geocoded CSLC along X-direction.')
    optional.add_argument('-dy', '--y-spac', type=float, default=10,
                          help='Spacing in meters of geocoded CSLC along Y-direction.')
    optional.add_argument('--bbox', nargs=4, type=float, default=None,
                          metavar=('xmin', 'ymin', 'xmax', 'ymax'),
                          help='Bounding box of the geocoded stack.')
    optional.add_argument('--epsg-bbox', type=int, default=4326,
                          help='EPSG code of the bounding box. '
                               'If 4326, the bounding box is in lon/lat degrees.')
    optional.add_argument('-e', '--epsg', type=int, default=None,
                          help='Output EPSG projection code for geocoded bursts. '
                               'If None, projection the UTM zone of the first burst.')
    optional.add_argument('-f', '--flatten', type=bool, default=True,
                          help='If True, enables topographic phase flattening.')
    optional.add_argument('-ss', '--range-split-spectrum',
                          dest='is_split_spectrum', type=bool, default=False,
                          help='If True, enables split-spectrum')
    optional.add_argument('-lb', '--low-band', type=float, default=0.0,
                          help='Low sub-band bandwidth in Hz (default: 0.0)')
    optional.add_argument('-hb', '--high-band', type=float, default=0.0,
                          help='High sub-band bandwidth in Hz (default: 0.0')
    return parser.parse_args()


def generate_burst_map(zip_files, orbit_dir, x_spac, y_spac, epsg=None,
                       bbox=None, epsg_bbox=4326):
    """Generates a dataframe of geogrid infos for each burst ID in `zip_files`.

    Parameters
    ----------
    zip_files: str
        List of S1-A/B SAFE (zip) files
    orbit_dir: str
        Directory containing sensor orbit ephemerides
    x_spac: float
        Spacing of geocoded burst along X-direction
    y_spac: float
        Spacing of geocoded burst along Y-direction
    epsg: int
        EPSG code identifying output product projection system
    bbox: Optional[tuple[float]]
        Desired bounding box of the geocoded bursts as (left, bottom, right, top).
        If not provided, the bounding box is computed for each burst.
    epsg_bbox: int
        EPSG code of the bounding box. If 4326, the bounding box is assumed
        to be lon/lat degrees (default: 4326).

    Returns
    -------
    burst_map: pandas.Dataframe
        Pandas dataframe containing geogrid info (e.g. top-left, bottom-right
        x and y coordinates) for each burst to process
    """
    # Initialize dictionary that contains all the info for geocoding
    burst_map = defaultdict(list)

    # Get all the bursts from safe file
    i_subswath = [1, 2, 3]

    for zip_file in zip_files:
        orbit_path = get_orbit_file_from_dir(zip_file, orbit_dir)

        for subswath in i_subswath:
            ref_bursts = load_bursts(zip_file, orbit_path, subswath)
            for burst in ref_bursts:
                if epsg is None:
                    epsg = get_point_epsg(burst.center.y,
                                          burst.center.x)

                if bbox is not None:
                    bbox_utm = helpers.bbox_to_utm(
                        bbox, epsg_bbox=epsg_bbox, epsg_out=epsg
                    )
                    burst_border_utm = helpers.polygon_to_utm(
                        burst.border[0], epsg
                    )
                    # Skip this burst if it doesn't intersect the specified bbox
                    if not geometry.box(*bbox_utm).intersects(burst_border_utm):
                        continue
                else:
                    # Initialize geogrid with the info checked at this stage
                    geogrid = isce3.product.bbox_to_geogrid(
                        burst.as_isce3_radargrid(),
                        burst.orbit,
                        isce3.core.LUT2d(),
                        x_spac, -y_spac,
                        epsg)
                    bbox_utm = snap_geogrid(geogrid, x_spac, y_spac)

                burst_map['burst_id'].append(burst.burst_id)
                # keep the burst object so we don't have to re-parse
                burst_map['burst'].append(burst)

                left, bottom, right, top = bbox_utm
                burst_map['x_top_left'].append(left)
                burst_map['y_top_left'].append(top)
                burst_map['x_bottom_right'].append(right)
                burst_map['y_bottom_right'].append(bottom)

                burst_map['epsg'].append(epsg)
                burst_map['date'].append(burst.sensing_start.strftime("%Y%m%d"))
                # Save the file paths for creating the runconfig
                burst_map['orbit_path'].append(orbit_path)
                burst_map['zip_file'].append(zip_file)

    burst_map = pd.DataFrame(data=burst_map)
    return burst_map


def snap_geogrid(geogrid, x_spac, y_spac):
    """Snap the bounding box of the geogrid to integer multiples of the
    spacing in X-/Y-directions.

    Parameters
    ----------
    geogrid : isce3.core.Geogrid
        Geogrid object
    x_spac : float
        Spacing of geocoded burst along X-direction
    y_spac : float
        Spacing of geocoded burst along Y-direction

    Returns
    -------
    tuple
        Tuple containing the snapped bounding box coordinates
        (left, bottom, right, top)
    """
    x_left = x_spac * np.floor(geogrid.start_x / x_spac)
    x_right = x_spac * np.ceil((geogrid.start_x + x_spac * geogrid.width) / x_spac)
    y_bottom = y_spac * np.floor((geogrid.start_y + y_spac * geogrid.length) / y_spac)
    y_top = y_spac * np.ceil(geogrid.start_y / y_spac)
    return (x_left, y_bottom, x_right, y_top)


def prune_dataframe(data, id_col, id_list):
    """Prune dataframe based on column ID and list of value

    Parameters:
    ----------
    data: pandas.DataFrame
        dataframe that needs to be pruned
    id_col: str
        column identification for 'data' (e.g. 'burst_id')
    id_list: list
        List of elements to consider when pruning.
        If exclude_items is False (default), then all elements in `data`
            will be kept *except for* those in `id_list`.
        If exclude_items is True, the items in `id_list` will be removed from `data`.

    Returns:
    -------
    data: pandas.DataFrame
        Pruned dataframe with rows in 'id_list'
    """
    pattern = '|'.join(id_list)
    df = data.loc[data[id_col].str.contains(pattern, case=False)]
    return df


def get_common_burst_ids(data):
    """Get list of burst IDs common among all processed dates

    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe containing info for stitching (e.g. burst IDs)

    Returns
    -------
    common_id: list
        List containing common burst IDs among all the dates
    """
    # Identify all the dates for the bursts to stitch
    unique_dates = list(set(data['date']))

    # Initialize list of unique burst IDs
    common_id = data.burst_id[data.date == unique_dates[0]]

    for date in unique_dates:
        ids = data.burst_id[data.date == date]
        common_id = sorted(list(set(ids.tolist()) & set(common_id)))
    return common_id


def create_runconfig(burst_map_row, dem_file, work_dir, flatten, enable_rss,
                     low_band, high_band, pol, x_spac, y_spac):
    """
    Create runconfig to process geocoded bursts

    Parameters
    ----------
    burst_map_row: namedtuple
        one row from the dataframe method `burst_map.itertuples()`
    dem_file: str
        Path to DEM to use for processing
    work_dir: str
        Path to working directory for temp and final results
    flatten: bool
        Flag to enable/disable flattening
    enable_rss: bool
        Flag to enable range split-spectrum
    low-band: float
        Low sub-image bandwidth (in Hz) for split-spectrum
    high-band: float
        High sub-image bandwidth (in Hz) for split-spectrum
    pol: str
        Polarizations to process: co-pol, dual-pol, cross-pol
    x_spac: float
        Spacing of geocoded burst along X-direction
    y_spac: float
        Spacing of geocoded burst along Y-direction

    Returns
    -------
    runconfig: str
        Path to runconfig file
    """
    # Load default runconfig and fill it with user-defined options
    yaml_path = f'{helpers.WORKFLOW_SCRIPTS_DIR}/defaults/s1_cslc_geo.yaml'
    with open(yaml_path, 'r') as stream:
        yaml_cfg = yaml.safe_load(stream)

    groups = yaml_cfg['runconfig']['groups']
    inputs = groups['input_file_group']
    product = groups['product_path_group']
    process = groups['processing']
    geocode = process['geocoding']
    rss = process['range_split_spectrum']

    # Allocate Inputs
    burst = burst_map_row.burst
    inputs['safe_file_path'] = [burst_map_row.zip_file]
    inputs['orbit_file_path'] = [burst_map_row.orbit_path]
    inputs['burst_id'] = burst.burst_id
    groups['dynamic_ancillary_file_group']['dem_file'] = dem_file

    # Product path
    product['product_path'] = work_dir
    product['scratch_path'] = f'{work_dir}/scratch'
    product['sas_output_file'] = work_dir

    # Geocoding
    process['polarization'] = pol
    geocode['flatten'] = flatten
    geocode['x_posting'] = x_spac
    geocode['y_posting'] = y_spac

    geocode['top_left']['x'] = burst_map_row.x_top_left
    geocode['top_left']['y'] = burst_map_row.y_top_left
    geocode['bottom_right']['x'] = burst_map_row.x_bottom_right
    geocode['bottom_right']['y'] = burst_map_row.y_bottom_right
    # geocode['x_snap'] = None
    # geocode['y_snap'] = None
    geocode['output_epsg'] = burst_map_row.epsg

    # Range split spectrum
    rss['enabled'] = enable_rss
    rss['low_band_bandwidth'] = low_band
    rss['high_band_bandwidth'] = high_band

    date_str = burst.sensing_start.strftime("%Y%m%d")
    os.makedirs(f'{work_dir}/runconfigs', exist_ok=True)
    runconfig_path = f'{work_dir}/runconfigs/geo_runconfig_{date_str}_{burst.burst_id}.yaml'
    with open(runconfig_path, 'w') as yaml_file:
        yaml.dump(yaml_cfg, yaml_file, default_flow_style=False)
    return runconfig_path


def _filter_by_date(zip_file_list, start_date, end_date, exclude_dates):
    """
    Filter list of zip files based on date

    Parameters
    ----------
    zip_file_list: list
        List of zip files to filter
    start_date: str
        Start date in YYYYMMDD format
    end_date: str
        End date in YYYYMMDD format
    exclude_dates: list
        List of dates to exclude

    Returns
    -------
    zip_file_list: list
        Filtered list of zip files
    """
    safe_datetimes = [parse_safe_filename(zip_file)[2] for zip_file in zip_file_list]
    if start_date:
        start_datetime = datetime.datetime.strptime(start_date, '%Y%m%d')
    else:
        start_datetime = min(safe_datetimes)
    if end_date:
        end_datetime = datetime.datetime.strptime(end_date, '%Y%m%d')
    else:
        end_datetime = max(safe_datetimes)

    if exclude_dates is not None:
        exclude_datetimes = [
            datetime.datetime.strptime(d, '%Y%m%d').date
            for d in exclude_dates
        ]
    else:
        exclude_datetimes = []

    # Filter within date range, and prune excluded dates
    zip_file_list = [
        f
        for (f, dt) in zip(zip_file_list, safe_datetimes)
        if start_datetime <= dt <= end_datetime and dt.date not in exclude_datetimes
    ]
    return zip_file_list


def run(slc_dir, dem_file, burst_id, start_date=None, end_date=None, exclude_dates=None,
        orbit_dir=None, work_dir='stack', pol='dual-pol', x_spac=5, y_spac=10, bbox=None,
        epsg_bbox=4326, epsg=None, flatten=True, is_split_spectrum=False, low_band=0.0,
        high_band=0.0):
    """Create runconfigs and runfiles generating geocoded bursts for a static
    stack of Sentinel-1 A/B SAFE files.

    Parameters
    ----------
    slc_dir: str
        Directory containing S1-A/B SAFE files
    dem_file: str
        File path to DEM to use for processing
    burst_id: list
        List of burst IDs to process (default: None)
    start_date: int
        Date of the start acquisition of the stack (format: YYYYMMDD)
    end_date: int
        Date of the end acquisition of the stack (format: YYYYMMDD)
    exclude_dates: list
        List of dates to exclude from the stack (format: YYYYMMDD)
    orbit_dir: str
        Directory containing orbit files
    work_dir: str
        Working directory to store temp and final files
    x_spac: float
        Spacing of geocoded burst along X-direction
    y_spac: float
        Spacing of geocoded burst along Y-direction
    bbox: tuple[float], optional
        Bounding box of the area to geocode: (xmin, ymin, xmax, ymax) in degrees.
        If not provided, will use the bounding box of the stack.
    epsg: int
        EPSG code identifying projection system to use for processing
    flatten: bool
        Enable/disable flattening of geocoded burst
    is_split_spectrum: bool
        Enable/disable range split spectrum
    low_band: float
        Low sub-band bandwidth for split-spectrum in Hz
    high_band: float
        High sub-band bandwidth for split-spectrum in Hz
    """
    start_time = time.time()
    error = journal.error('s1_geo_stack_processor.main')
    info = journal.info('s1_geo_stack_processor.main')

    # Check if SLC dir and DEM exists
    if not os.path.isdir(slc_dir):
        err_str = f'{slc_dir} SLC directory does not exist'
        error.log(err_str)
        raise FileNotFoundError(err_str)

    if not os.path.isfile(dem_file):
        err_str = f'{dem_file} DEM file does not exists'
        error.log(err_str)
        raise FileNotFoundError(err_str)

    # Create directory for runfiles
    run_dir = f'{work_dir}/run_files'
    os.makedirs(run_dir, exist_ok=True)

    # Check if orbit are provided, if Not download
    if orbit_dir is None:
        orbit_dir = f'{work_dir}/orbits'
        info.log(f'Orbit directory not assigned. Using {orbit_dir} to download orbits')
        os.makedirs(orbit_dir, exist_ok=True)
        # Note: Specific files will be downloaded as needed during `generate_burst_map`

    # Generate burst map and prune it if a list of burst ID is provided
    zip_file_list = sorted(glob.glob(f'{slc_dir}/S1*zip'))
    # Remove zip files that are not in the date range before generating burst map
    zip_file_list = _filter_by_date(zip_file_list, start_date, end_date, exclude_dates)

    info.log(f'Generating burst map for {len(zip_file_list)} SAFE files')
    burst_map = generate_burst_map(zip_file_list, orbit_dir, x_spac, y_spac, epsg, bbox)

    # Identify burst IDs common across the stack and remove from the dataframe
    # burst IDs that are not in common
    common_ids = get_common_burst_ids(burst_map)
    burst_map = prune_dataframe(burst_map, 'burst_id', common_ids)

    # If user selects burst IDs to process, prune unnecessary bursts
    if burst_id is not None:
        burst_map = prune_dataframe(burst_map, 'burst_id', burst_id)

    # Ready to geocode bursts
    for row in burst_map.itertuples():
        runconfig_path = create_runconfig(
            row,
            dem_file,
            work_dir,
            flatten,
            is_split_spectrum,
            low_band,
            high_band,
            pol,
            x_spac,
            y_spac
        )
        date_str = row.burst.sensing_start.strftime("%Y%m%d")
        runfile_name = f'{run_dir}/run_{date_str}_{row.burst.burst_id}.sh'
        with open(runfile_name, 'w') as rsh:
            path = os.path.dirname(os.path.realpath(__file__))
            rsh.write(
                f'python {path}/s1_cslc.py {runconfig_path}\n')

    end_time = time.time()
    print('Elapsed time (min):', (end_time - start_time) / 60.0)


def main():
    """Create the command line interface and run the script."""
    # Run main script
    args = create_parser()

    run(args.slc_dir, args.dem_file, args.burst_id, args.start_date,
        args.end_date, args.exclude_dates, args.orbit_dir,
        args.work_dir, args.pol, args.x_spac, args.y_spac, args.bbox,
        args.epsg_bbox, args.epsg, args.flatten, args.is_split_spectrum,
        args.low_band, args.high_band)


if __name__ == '__main__':
    main()
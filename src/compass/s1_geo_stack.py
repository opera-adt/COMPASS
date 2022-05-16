import argparse
import os, sys, glob
import journal
import cgi
import requests
import datetime
import pandas as pd
import yaml
import isce3
import numpy as np

from xml.etree import ElementTree
from osgeo import osr
from s1reader.s1_reader import load_bursts
from s1reader.s1_orbit import get_orbit_file_from_list
from compass.utils import helpers
from compass.utils.runconfig import load_validate_yaml

from compass.utils.geo_grid import get_point_epsg
from compass.utils.geo_runconfig import GeoRunConfig

# Required for orbit download
scihub_url = 'https://scihub.copernicus.eu/gnss/odata/v1/Products'
# Namespaces of the XML file returned by the S1 query. Will they change it?
w3_url = '{http://www.w3.org/2005/Atom}'
m_url = '{http://schemas.microsoft.com/ado/2007/08/dataservices/metadata}'
d_url = '{http://schemas.microsoft.com/ado/2007/08/dataservices}'
# Scihub guest credential
scihub_user = 'gnssguest'
scihub_password = 'gnssguest'


def create_parser():
    parser = argparse.ArgumentParser(
        description='S1-A/B geocoded CSLC stack processor.')
    parser.add_argument('-s', '--slc-dir', dest='slc_dir', type=str,
                        required=True,
                        help='Directory containing the S1-A/B SLCs (zip files)')
    parser.add_argument('-d', '--dem-file', dest='dem_file', type=str,
                        required=True,
                        help='File path to DEM to use for processing.')
    parser.add_argument('-o', '--orbit-dir', dest='orbit_dir', type=str,
                        default=None,
                        help='Directory with orbit files. If None, downloads orbit files')
    parser.add_argument('-w', '--working-dir', dest='work_dir', type=str,
                        default='stack',
                        help='Directory to store intermediate and final results')
    parser.add_argument('-r', '--ref-date', dest='ref_date', type=int,
                        default=None,
                        help='Date of reference acquisition (yyyymmdd). If None, first acquisition of '
                             'the stack is selected as reference.')
    parser.add_argument('-b', '--burst-id', dest='burst_id', nargs='+',
                        default=None,
                        help='List of burst IDs to process. If None, all the burst IDs in the'
                             'reference date are processed. (default: None)')
    parser.add_argument('-p', '--pol', dest='pol', nargs='+', default='co-pol',
                        help='Polarization to process: dual-pol, co-pol, cross-pol (default: co-pol).')
    parser.add_argument('-x', '--x-spac', dest='x_spac', type=float, default=5,
                        help='Spacing in meters of geocoded CSLC along X-direction.')
    parser.add_argument('-y', '--y-spac', dest='y_spac', type=float, default=10,
                        help='Spacing in meters of geocoded CSLC along Y-direction.')
    parser.add_argument('-e', '--epsg', dest='epsg', type=int, default=None,
                        help='EPSG projection code for output geocoded bursts')
    parser.add_argument('-f', '--flatten', dest='flatten', type=bool,
                        default=True,
                        help='If True, enables flattening (default: True)')
    parser.add_argument('-ss', '--range-split-spectrum',
                        dest='is_split_spectrum', type=bool, default=False,
                        help='If True, enables split-spectrum (default: False)')
    parser.add_argument('-lb', '--low-band', dest='low_band', type=float,
                        default=0.0,
                        help='Low sub-band bandwidth in Hz (default: 0.0)')
    parser.add_argument('-hb', '--high-band', dest='high_band', type=float,
                        default=0.0,
                        help='High sub-band bandwidth in Hz (default: 0.0')
    return parser.parse_args()


def check_internet_connection():
    '''
    Check connection availability
    '''
    url = "http://google.com"
    try:
        request = requests.get(url, timeout=10)
    except (requests.ConnectionError, requests.Timeout) as exception:
        raise sys.exit(exception)


def parse_safe_filename(safe_filename):
    '''
    Extract info from S1-A/B SAFE filename
    SAFE filename structure: S1A_IW_SLC__1SDV_20150224T114043_20150224T114111_004764_005E86_AD02.SAFE

    Parameters:
    -----------
    safe_filename: string
       Path to S1-A/B SAFE file

    Returns:
    -------
    List of [sensor_id, mode_id, start_datetime,
                end_datetime, abs_orbit_num]
       sensor_id: sensor identifier (S1A or S1B)
       mode_id: mode/beam (e.g. IW)
       start_datetime: acquisition start datetime
       stop_datetime: acquisition stop datetime
       abs_orbit_num: absolute orbit number
    '''

    safe_name = os.path.basename(safe_filename)
    sensor_id = safe_name[2]
    sensor_mode = safe_name[4:10]
    start_datetime = datetime.datetime.strptime(safe_name[17:32],
                                                '%Y%m%dT%H%M%S')
    end_datetime = datetime.datetime.strptime(safe_name[33:48],
                                              '%Y%m%dT%H%M%S')
    abs_orb_num = int(safe_name[49:55])

    return [sensor_id, sensor_mode, start_datetime, end_datetime, abs_orb_num]


def get_orbit_dict(sensor_id, start_time, end_time, orbit_type):
    '''
    Query Copernicus GNSS API to find latest orbit file

    Parameters:
    ----------
    sensor_id: str
        Sentinel satellite identifier ('A' or 'B')
    start_time: datetime object
        Sentinel start acquisition time
    end_time: datetime object
        Sentinel end acquisition time
    orbit_type: str
        Type of orbit to download (AUX_POEORB: precise, AUX_RESORB: restituted)

    Returns:
    orbit_dict: dict
        Python dictionary with [orbit_name, orbit_type, download_url]
    '''
    # Check if correct orbit_type
    if orbit_type not in ['AUX_POEORB', 'AUX_RESORB']:
        err_msg = f'{orbit_type} not a valid orbit type'
        raise ValueError(err_msg)

    # Add a 30 min margin to start_time and end_time
    pad_start_time = start_time - datetime.timedelta(hours=0.5)
    pad_end_time = end_time + datetime.timedelta(hours=0.5)
    new_start_time = pad_start_time.strftime('%Y-%m-%dT%H:%M:%S')
    new_end_time = pad_end_time.strftime('%Y-%m-%dT%H:%M:%S')
    query_string = f"startswith(Name,'S1{sensor_id}') and substringof('{orbit_type}',Name) " \
                   f"and ContentDate/Start lt datetime'{new_start_time}' and ContentDate/End gt datetime'{new_end_time}'"
    query_params = {'$top': 1, '$orderby': 'ContentDate/Start asc',
                    '$filter': query_string}
    query_response = requests.get(url=scihub_url, params=query_params,
                                  auth=(scihub_user, scihub_password))
    # Parse XML tree from query response
    xml_tree = ElementTree.fromstring(query_response.content)
    # Extract w3.org URL
    w3_url = xml_tree.tag.split('feed')[0]

    # Extract orbit's name, id, url
    orbit_id = xml_tree.findtext(
        f'.//{w3_url}entry/{m_url}properties/{d_url}Id')
    orbit_url = f"{scihub_url}('{orbit_id}')/$value"
    orbit_name = xml_tree.findtext(f'./{w3_url}entry/{w3_url}title')

    if orbit_id is not None:
        orbit_dict = {'orbit_name': orbit_name, 'orbit_type': orbit_type,
                      'orbit_url': orbit_url}
    else:
        orbit_dict = None
    return orbit_dict


def download_orbit(output_folder, orbit_url):
    '''
    Download S1-A/B orbits

    Parameters:
    ----------
    output_folder: str
        Path to directory where to store orbits
    orbit_url: str
        Remote url of orbit file to download
    '''

    response = requests.get(url=orbit_url, auth=(scihub_user, scihub_password))
    # Get header and find filename
    header = response.headers['content-disposition']
    header_value, header_params = cgi.parse_header(header)
    # construct orbit filename
    orbit_filename = os.path.join(output_folder, header_params['filename'])
    # Save orbits
    open(orbit_filename, 'wb').write(response.content)


def generate_burst_map(ref_file, orbit_dir, x_spac, y_spac, epsg=4326):
    '''
    Generates dataframe containing geogrid info for each burst ID in
    the ref_file

    Parameters
    ----------
    ref_file: str
        File path to the stack reference file
    orbit_dir: str
        Directory containing sensor orbit ephemerides
    x_spac: float
        Spacing of geocoded burst along X-direction
    y_spac: float
        Spacing of geocoded burst along Y-direction
    epsg: int
        EPSG code identifying output product projection system

    Returns
    -------
    burst_map: pandas.Dataframe
        Pandas dataframe containing geogrid info (e.g. top-left, bottom-right
        x and y coordinates) for each burst to process
    '''
    # Initialize dictionary that contains all the info for geocoding
    burst_map = {'burst_id': [], 'x_top_left': [], 'y_top_left': [],
                 'x_bottom_right': [], 'y_bottom_right': [], 'epsg': []}

    # Get all the bursts from safe file
    i_subswath = [1, 2, 3]
    orbit_path = get_orbit_file_from_list(ref_file,
                                          glob.glob(f'{orbit_dir}/S1*'))

    for subswath in i_subswath:
        ref_bursts = load_bursts(ref_file, orbit_path, subswath)
        for burst in ref_bursts:
            burst_map['burst_id'].append(burst.burst_id)

            # TO DO: Correct when integrated to compass
            if epsg is None:
                epsg = get_point_epsg(burst.center.y,
                                      burst.center.x)

            # Initialize geogrid with the info checked at this stage
            geogrid = isce3.product.bbox_to_geogrid(burst.as_isce3_radargrid(),
                                                    burst.orbit,
                                                    isce3.core.LUT2d(),
                                                    x_spac, -y_spac,
                                                    epsg)

            # Snap coordinates so that adjacent burst coordinates are integer
            # multiples of the spacing in X-/Y-directions
            burst_map['x_top_left'].append(
                x_spac * np.floor(geogrid.start_x / x_spac))
            burst_map['y_top_left'].append(
                y_spac * np.ceil(geogrid.start_y / y_spac))
            burst_map['x_bottom_right'].append(
                x_spac * np.ceil(
                    (geogrid.start_x + x_spac * geogrid.width) / x_spac))
            burst_map['y_bottom_right'].append(
                y_spac * np.floor(
                    (geogrid.start_y + y_spac * geogrid.length) / y_spac))
            burst_map['epsg'].append(epsg)

    map = pd.DataFrame(data=burst_map)
    return map


def prune_dataframe(data, id_col, id_list):
    '''
    Prune dataframe based on column ID and list of value
    Parameters:
    ----------
    data: pandas.DataFrame
       dataframe that needs to be pruned
    id_col: str
       column identification for 'data' (e.g. 'burst_id')
    id_list: list
       List of elements to keep after pruning. Elements not
       in id_list but contained in 'data' will be pruned
    Returns:
    -------
    data: pandas.DataFrame
       Pruned dataframe with rows in 'id_list'
    '''
    pattern = '|'.join(id_list)
    dataf = data.loc[data[id_col].str.contains(pattern,
                                               case=False)]
    return dataf


def create_runconfig(burst, safe, orbit_path, dem_file, work_dir,
                     burst_map, flatten, enable_rss, low_band, high_band, pol,
                     x_spac, y_spac):
    '''
    Create runconfig to process geocoded bursts

    Parameters
    ---------
    burst: Sentinel1BurstSlc
        Object containing info on burst to process
    safe: str
        Path to SAFE file containing burst to process
    orbit_path: str
        Path to orbit file related to burst to process
    dem_file: str
        Path to DEM to use for processing
    work_dir: str
        Path to working directory for temp and final results
    burst_map: pandas.Dataframe
        Pandas dataframe containing burst top-left, bottom-right coordinates
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
    '''
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
    inputs['safe_file_path'] = [safe]
    inputs['orbit_file_path'] = [orbit_path]
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
    geocode['top_left']['x'] = \
    burst_map.x_top_left[burst_map.burst_id == burst.burst_id].tolist()[0]
    geocode['top_left']['y'] = \
    burst_map.y_top_left[burst_map.burst_id == burst.burst_id].tolist()[0]
    geocode['bottom_right']['x'] = \
    burst_map.x_bottom_right[burst_map.burst_id == burst.burst_id].tolist()[0]
    geocode['bottom_right']['y'] = \
    burst_map.y_bottom_right[burst_map.burst_id == burst.burst_id].tolist()[0]
    # geocode['x_snap'] = None
    # geocode['y_snap'] = None
    geocode['output_epsg'] = \
    burst_map.epsg[burst_map.burst_id == burst.burst_id].tolist()[0]

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


def main(slc_dir, dem_file, burst_id, ref_date=None, orbit_dir=None,
         work_dir='stack',
         pol='dual-pol', x_spac=5, y_spac=10, epsg=None,
         flatten= True,
         is_split_spectrum=False,
         low_band=0.0,
         high_band=0.0):
    '''
    Create runconfigs and runfiles generating geocoded bursts for
    a static stack of Sentinel-1 A/B SAFE files.

    Parameters
    ----------
    slc_dir: str
        Directory containing S1-A/B SAFE files
    dem_file: str
        File path to DEM to use for processing
    burst_id: list
        List of burst IDs to process (default: None)
    ref_date: str
        Date of reference acquisition of the stack (format: YYYYMMDD)
    orbit_dir: str
        Directory containing orbit files
    work_dir: str
        Working directory to store temp and final files
    x_spac: float
        Spacing of geocoded burst along X-direction
    y_spac: float
        Spacing of geocoded burst along Y-direction
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
    '''
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
        info.log('Orbit directory not assigned. Download orbit ephemerides')

        # Create orbit dir and check internet connection
        orbit_dir = f'{work_dir}/orbits'
        os.makedirs(orbit_dir, exist_ok=True)
        check_internet_connection()

        # List all zip file and extract info
        zip_file_list = sorted(glob.glob(f'{slc_dir}/S1*zip'))

        for zip_file in zip_file_list:
            sensor_id, _, start_datetime, \
            end_datetime, _ = parse_safe_filename(zip_file)

            # Find precise orbits first
            orbit_dict = get_orbit_dict(sensor_id, start_datetime,
                                        end_datetime, 'AUX_POEORB')
            # If orbit_dict is empty, precise orbits have not been found. Find restituted orbits instead
            if orbit_dict == None:
                orbit_dict = get_orbit_dict(sensor_id, start_datetime,
                                            end_datetime, 'AUX_RESORB')
            # Download the orbit file
            download_orbit(orbit_dir, orbit_dict['orbit_url'])

    # Find reference date and construct dict for geocoding
    if ref_date is None:
        ref_file = sorted(glob.glob(f'{slc_dir}/S1*zip'))[0]
    else:
        ref_file = glob.glob(f'{slc_dir}/S1*{str(ref_date)}*zip')[0]

    # Generate burst map and prune it if a list of burst ID is provided
    burst_map = generate_burst_map(ref_file, orbit_dir, x_spac, y_spac, epsg)

    if burst_id is not None:
        burst_map = prune_dataframe(burst_map, 'burst_id', burst_id)

    # Start to geocode bursts
    zip_file = sorted(glob.glob(f'{slc_dir}/S1*zip'))
    for safe in zip_file:
        i_subswath = [1, 2, 3]
        orbit_path = get_orbit_file_from_list(safe,
                                              glob.glob(f'{orbit_dir}/S1*'))

        for subswath in i_subswath:
            bursts = load_bursts(safe, orbit_path, subswath)
            for burst in bursts:
                if burst.burst_id in list(set(burst_map['burst_id'])):
                    runconfig_path = create_runconfig(burst, safe, orbit_path,
                                                      dem_file,
                                                      work_dir, burst_map, flatten,
                                                      is_split_spectrum,
                                                      low_band, high_band, pol,
                                                      x_spac, y_spac)
                    date_str = burst.sensing_start.strftime("%Y%m%d")
                    runfile_name = f'{run_dir}/run_{date_str}_{burst.burst_id}.sh'
                    with open(runfile_name, 'w') as rsh:
                        path = os.path.dirname(os.path.realpath(__file__))
                        rsh.write(
                            f'python {path}/geo_cslc.py {runconfig_path}\n')
                else:
                    info.log(f'{burst.burst_id} not part of the stack')


if __name__ == '__main__':
    # Run main script
    args = create_parser()

    main(args.slc_dir, args.dem_file, args.burst_id, args.ref_date,
         args.orbit_dir,
         args.work_dir, args.pol, args.x_spac, args.y_spac, args.epsg,
         args.flatten, args.is_split_spectrum,
         args.low_band, args.high_band)

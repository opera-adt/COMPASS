'''
Collection of functions to help write HDF5 datasets and metadata
'''

from dataclasses import dataclass, field
from datetime import datetime
import os

import isce3
import numpy as np
from osgeo import osr, gdal
import s1reader
from s1reader.s1_burst_slc import Sentinel1BurstSlc
import shapely

import compass
from compass.utils.fill_value import determine_fill_value


TIME_STR_FMT = '%Y-%m-%d %H:%M:%S.%f'
ROOT_PATH = '/'
DATA_PATH = '/data'
QA_PATH = '/quality_assurance'
METADATA_PATH = '/metadata'

@dataclass
class Meta:
    '''
    Convenience dataclass for passing parameters to be written to h5py.Dataset
    '''
    # Dataset name
    name: str
    # Data to be stored in Dataset
    value: object
    # Description attribute of Dataset
    description: str
    # Other attributes to be written to Dataset
    attr_dict: dict = field(default_factory=dict)


def _as_np_string_if_needed(val):
    '''
    If type str encountered, convert and return as np.string_. Otherwise return
    as is.
    '''
    val = np.string_(val) if isinstance(val, str) else val
    return val


def add_dataset_and_attrs(group, meta_item):
    '''Write isce3.core.Poly1d properties to hdf5

    Parameters
    ----------
    group: h5py.Group
        h5py Group to store poly1d parameters in
    meta_item: Meta
        Name of dataset to add
    '''
    # Ensure it is clear to write by deleting pre-existing Dataset
    if meta_item.name in group:
        del group[meta_item.name]

    # Convert data to written if necessary
    val = _as_np_string_if_needed(meta_item.value)
    try:
        if val is None:
            group[meta_item.name] = np.nan
        else:
            group[meta_item.name] = val
    except TypeError:
        raise TypeError(f'unable to write {meta_item.name}')

    # Write data and attributes
    val_ds = group[meta_item.name]
    desc = _as_np_string_if_needed(meta_item.description)
    val_ds.attrs['description'] = desc
    for key, val in meta_item.attr_dict.items():
        val_ds.attrs[key] = _as_np_string_if_needed(val)


def init_geocoded_dataset(data_group, dataset_name, geo_grid, dtype,
                          description, data=None, output_cfg=None,
                          fill_val=None):
    '''
    Create and allocate dataset for isce.geocode.geocode_slc to write to that
    is CF-compliant. If data parameter not provided, then an appropriate fill
    value is found and used to fill dataset.

    Parameters
    ----------
    data_group: h5py.Group
        h5py group where geocoded dataset will be created in
    dataset_name: str
        Name of dataset to be created
    geo_grid: isce3.product.GeoGridParameters
        Geogrid of output
    dtype: Union(str, type)
        Data type of dataset to be geocoded to be passed to require_dataset.
        require_dataset can take string values e.g. "float32" or types e.g.
        numpy.float32
    description: str
        Description of dataset to be geocoded
    data: np.ndarray
        Array to set dataset raster with
    output_cfg: dict
        Optional dict containing output options in runconfig to apply to
        created datasets
    fill_val: float
        Optional value to fill an empty dataset

    Returns
    -------
    cslc_ds: h5py.Dataset
        NCDF compliant h5py dataset ready to be populated with geocoded raster
    '''
    # Default to no dataset keyword args
    output_kwargs = {}

    # Always set chunks kwarg
    output_kwargs['chunks'] = tuple(output_cfg.chunk_size)

    # If compression is enabled, populate kwargs from runconfig contents
    if output_cfg.compression_enabled:
        output_kwargs['compression'] = 'gzip'
        output_kwargs['compression_opts'] = output_cfg.compression_level
        output_kwargs['shuffle'] = output_cfg.shuffle

    # Shape of dataset is defined by the geo grid
    shape = (geo_grid.length, geo_grid.width)

    # Determine fill value of dataset to either correctly init empty dataset
    # and/or populate dataset attribute
    _fill_val = determine_fill_value(dtype, fill_val)

    # If data is None, create dataset to specified parameters and fill with
    # specified fill value. If data is not None, create a dataset with
    # provided data.
    if data is None:
        # Create a dataset with shape and a fill value from above
        cslc_ds = data_group.require_dataset(dataset_name, dtype=dtype,
                                             shape=shape, fillvalue=_fill_val,
                                             **output_kwargs)
    else:
        # Create a dataset with provided data
        cslc_ds = data_group.create_dataset(dataset_name, data=data,
                                            **output_kwargs)

    cslc_ds.attrs['description'] = description
    cslc_ds.attrs['fill_value'] = _fill_val

    # Compute x scale
    dx = geo_grid.spacing_x
    x0 = geo_grid.start_x + 0.5 * dx
    xf = x0 + (geo_grid.width - 1) * dx
    x_vect = np.linspace(x0, xf, geo_grid.width, dtype=np.float64)

    # Compute y scale
    dy = geo_grid.spacing_y
    y0 = geo_grid.start_y + 0.5 * dy
    yf = y0 + (geo_grid.length - 1) * dy
    y_vect = np.linspace(y0, yf, geo_grid.length, dtype=np.float64)

    # following copied and pasted (and slightly modified) from:
    # https://github-fn.jpl.nasa.gov/isce-3/isce/wiki/CF-Conventions-and-Map-Projections
    x_ds = data_group.require_dataset('x_coordinates', dtype='float64',
                                      data=x_vect, shape=x_vect.shape)
    y_ds = data_group.require_dataset('y_coordinates', dtype='float64',
                                      data=y_vect, shape=y_vect.shape)

    # Mapping of dimension scales to datasets is not done automatically in HDF5
    # We should label appropriate arrays as scales and attach them to datasets
    # explicitly as show below.
    x_ds.make_scale()
    cslc_ds.dims[1].attach_scale(x_ds)
    y_ds.make_scale()
    cslc_ds.dims[0].attach_scale(y_ds)

    # Associate grid mapping with data - projection created later
    cslc_ds.attrs['grid_mapping'] = np.string_("projection")

    # Build list of metadata to be inserted to accompany dataset
    grid_meta_items = [
        Meta('x_spacing', geo_grid.spacing_x,
             'Spacing of the geographical grid along X-direction',
             {'units': 'meters'}),
        Meta('y_spacing', geo_grid.spacing_y,
             'Spacing of the geographical grid along Y-direction',
             {'units': 'meters'})
    ]
    for meta_item in grid_meta_items:
        add_dataset_and_attrs(data_group, meta_item)

    # Set up osr for wkt
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(geo_grid.epsg)

    #Create a new single int dataset for projections
    projection_ds = data_group.require_dataset('projection', (), dtype='i')
    projection_ds[()] = geo_grid.epsg

    # Add description as an attribute to projection
    projection_ds.attrs['description'] = np.string_("Projection system")

    # WGS84 ellipsoid
    projection_ds.attrs['semi_major_axis'] = 6378137.0
    projection_ds.attrs['inverse_flattening'] = 298.257223563
    projection_ds.attrs['ellipsoid'] = np.string_("WGS84")

    # Additional fields
    projection_ds.attrs['epsg_code'] = geo_grid.epsg

    # CF 1.7+ requires this attribute to be named "crs_wkt"
    # spatial_ref is old GDAL way. Using that for testing only.
    # For NISAR replace with "crs_wkt"
    projection_ds.attrs['spatial_ref'] = np.string_(srs.ExportToWkt())

    # Here we have handcoded the attributes for the different cases
    # Recommended method is to use pyproj.CRS.to_cf() as shown above
    # To get complete set of attributes.

    # Geodetic latitude / longitude
    if geo_grid.epsg == 4326:
        #Set up grid mapping
        projection_ds.attrs['grid_mapping_name'] = np.string_('latitude_longitude')
        projection_ds.attrs['longitude_of_prime_meridian'] = 0.0

        #Setup units for x and y
        x_ds.attrs['standard_name'] = np.string_("longitude")
        x_ds.attrs['units'] = np.string_("degrees_east")

        y_ds.attrs['standard_name'] = np.string_("latitude")
        y_ds.attrs['units'] = np.string_("degrees_north")

    # UTM zones
    elif (geo_grid.epsg > 32600 and geo_grid.epsg < 32661) or \
         (geo_grid.epsg > 32700 and geo_grid.epsg < 32761):
        #Set up grid mapping
        projection_ds.attrs['grid_mapping_name'] = np.string_('universal_transverse_mercator')
        projection_ds.attrs['utm_zone_number'] = geo_grid.epsg % 100

        #Setup units for x and y
        x_ds.attrs['description'] = np.string_("CF compliant dimension associated with the X coordinate")
        x_ds.attrs['standard_name'] = np.string_("projection_x_coordinate")
        x_ds.attrs['long_name'] = np.string_("x coordinate of projection")
        x_ds.attrs['units'] = np.string_("meters")

        y_ds.attrs['description'] = np.string_("CF compliant dimension associated with the Y coordinate")
        y_ds.attrs['standard_name'] = np.string_("projection_y_coordinate")
        y_ds.attrs['long_name'] = np.string_("y coordinate of projection")
        y_ds.attrs['units'] = np.string_("meters")

    # Polar Stereo North
    elif geo_grid.epsg == 3413:
        #Set up grid mapping
        projection_ds.attrs['grid_mapping_name'] = np.string_("polar_stereographic")
        projection_ds.attrs['latitude_of_projection_origin'] = 90.0
        projection_ds.attrs['standard_parallel'] = 70.0
        projection_ds.attrs['straight_vertical_longitude_from_pole'] = -45.0
        projection_ds.attrs['false_easting'] = 0.0
        projection_ds.attrs['false_northing'] = 0.0

        #Setup units for x and y
        x_ds.attrs['standard_name'] = np.string_("projection_x_coordinate")
        x_ds.attrs['long_name'] = np.string_("x coordinate of projection")
        x_ds.attrs['units'] = np.string_("m")

        y_ds.attrs['standard_name'] = np.string_("projection_y_coordinate")
        y_ds.attrs['long_name'] = np.string_("y coordinate of projection")
        y_ds.attrs['units'] = np.string_("m")

    # Polar Stereo south
    elif geo_grid.epsg == 3031:
        #Set up grid mapping
        projection_ds.attrs['grid_mapping_name'] = np.string_("polar_stereographic")
        projection_ds.attrs['latitude_of_projection_origin'] = -90.0
        projection_ds.attrs['standard_parallel'] = -71.0
        projection_ds.attrs['straight_vertical_longitude_from_pole'] = 0.0
        projection_ds.attrs['false_easting'] = 0.0
        projection_ds.attrs['false_northing'] = 0.0

        #Setup units for x and y
        x_ds.attrs['standard_name'] = np.string_("projection_x_coordinate")
        x_ds.attrs['long_name'] = np.string_("x coordinate of projection")
        x_ds.attrs['units'] = np.string_("m")

        y_ds.attrs['standard_name'] = np.string_("projection_y_coordinate")
        y_ds.attrs['long_name'] = np.string_("y coordinate of projection")
        y_ds.attrs['units'] = np.string_("m")

    # EASE 2 for soil moisture L3
    elif geo_grid.epsg == 6933:
        #Set up grid mapping
        projection_ds.attrs['grid_mapping_name'] = np.string_("lambert_cylindrical_equal_area")
        projection_ds.attrs['longitude_of_central_meridian'] = 0.0
        projection_ds.attrs['standard_parallel'] = 30.0
        projection_ds.attrs['false_easting'] = 0.0
        projection_ds.attrs['false_northing'] = 0.0

        #Setup units for x and y
        x_ds.attrs['standard_name'] = np.string_("projection_x_coordinate")
        x_ds.attrs['long_name'] = np.string_("x coordinate of projection")
        x_ds.attrs['units'] = np.string_("m")

        y_ds.attrs['standard_name'] = np.string_("projection_y_coordinate")
        y_ds.attrs['long_name'] = np.string_("y coordinate of projection")
        y_ds.attrs['units'] = np.string_("m")

    # Europe Equal Area for Deformation map (to be implemented in isce3)
    elif geo_grid.epsg == 3035:
        #Set up grid mapping
        projection_ds.attrs['grid_mapping_name'] = np.string_("lambert_azimuthal_equal_area")
        projection_ds.attrs['longitude_of_projection_origin']= 10.0
        projection_ds.attrs['latitude_of_projection_origin'] = 52.0
        projection_ds.attrs['standard_parallel'] = -71.0
        projection_ds.attrs['straight_vertical_longitude_from_pole'] = 0.0
        projection_ds.attrs['false_easting'] = 4321000.0
        projection_ds.attrs['false_northing'] = 3210000.0

        #Setup units for x and y
        x_ds.attrs['standard_name'] = np.string_("projection_x_coordinate")
        x_ds.attrs['long_name'] = np.string_("x coordinate of projection")
        x_ds.attrs['units'] = np.string_("m")

        y_ds.attrs['standard_name'] = np.string_("projection_y_coordinate")
        y_ds.attrs['long_name'] = np.string_("y coordinate of projection")
        y_ds.attrs['units'] = np.string_("m")

    else:
        raise NotImplementedError('Waiting for implementation / Not supported in ISCE3')

    return cslc_ds


def save_orbit(orbit, orbit_direction, orbit_type, orbit_group):
    '''
    Write burst to HDF5

    Parameter
    ---------
    orbit: isce3.core.Orbit
        ISCE3 orbit object
    orbit_direction: string
        Orbit direction: ascending or descending
    orbit_type: string
        Type of orbit: RESORB or POEORB
    orbit_group: h5py.Group
        HDF5 group where orbit parameters will be written
    '''
    # isce isoformat gives 9 decimal places, but python `fromisoformat` wants 6
    ref_epoch = orbit.reference_epoch.isoformat().replace('T', ' ')[:-3]
    orbit_items = [
        Meta('reference_epoch', ref_epoch, 'Reference epoch of the state vectors',
             {'format': 'YYYY-MM-DD HH:MM:SS.6f'}),
        Meta('time', np.linspace(orbit.time.first,
                                 orbit.time.last,
                                 orbit.time.size),
             'Time of the orbit state vectors relative to the reference epoch',
             {'units': 'seconds'}),
        Meta('orbit_direction', orbit_direction,
             'Direction of sensor orbit ephemeris (e.g., ascending, descending)')
    ]
    for i_ax, axis in enumerate('xyz'):
        desc_suffix = f'{axis}-direction with respect to WGS84 G1762 reference frame'
        orbit_items.append(Meta(f'position_{axis}', orbit.position[:, i_ax],
                                f'Platform position along {desc_suffix}',
                                {'units': 'meters'}))
        orbit_items.append(Meta(f'velocity_{axis}', orbit.velocity[:, i_ax],
                                f'Platform velocity along {desc_suffix}',
                                {'units': 'meters/second'}))
    for meta_item in orbit_items:
        add_dataset_and_attrs(orbit_group, meta_item)

    orbit_ds = orbit_group.require_dataset("orbit_type", (), "S10",
                                           data=np.string_(orbit_type))
    orbit_ds.attrs["description"] = np.string_("Type of orbit file used for processing. "
                                               "RESORB: restituted orbit ephemeris or POEORB: precise orbit ephemeris")


def get_polygon_wkt(burst: Sentinel1BurstSlc):
    '''
    Get WKT for butst's bounding polygon
    It returns "POLYGON" when
    there is only one polygon that defines the burst's border
    It returns "MULTIPOLYGON" when
    there is more than one polygon that defines the burst's border
    Parameters:
    -----------
    burst: Sentinel1BurstSlc
        Input burst
    Return:
    _ : str
        "POLYGON" or "MULTIPOLYGON" in WKT
        as the bounding polygon of the input burst

    '''

    if len(burst.border) ==1:
        geometry_polygon = burst.border[0]
    else:
        geometry_polygon = shapely.geometry.MultiPolygon(burst.border)

    return geometry_polygon.wkt


def identity_to_h5group(dst_group, burst, cfg, product_type):
    '''
    Write burst metadata to HDF5

    Parameter:
    ---------
    dst_group: h5py Group
        HDF5 group metadata will be written to
    burst: Sentinel1BurstSlc
        Burst whose metadata is to written to HDF5
    cfg: dict[Namespace]
        Name space dictionary with runconfig parameters
    product_type: str
        Type of COMPASS product

    '''
    # identification datasets
    id_meta_items = [
        Meta('product_version', f'{cfg.product_group.product_version}', 'CSLC-S1 product version'),
        Meta('product_specification_version', f'{cfg.product_group.product_specification_version}',
             'CSLC-S1 product specification version'),
        Meta('absolute_orbit_number', burst.abs_orbit_number, 'Absolute orbit number'),
        Meta('track_number', burst.burst_id.track_number, 'Track number',
             {'units': 'unitless'}),
        Meta('burst_id', str(burst.burst_id), 'Burst identification string (burst ID)'),
        Meta('bounding_polygon', get_polygon_wkt(burst),
             'OGR compatible WKT representation of bounding polygon of the image',
             {'units':'degrees'}),
        Meta('mission_id', burst.platform_id, 'Mission identifier'),
        Meta('processing_date_time', datetime.now().strftime(TIME_STR_FMT),
             'Data processing date and time'),
        Meta('product_type', product_type, 'Product type'),
        Meta('product_level', 'L2', 'L0A: Unprocessed instrument data; L0B: Reformatted, '
             'unprocessed instrument data; L1: Processed instrument data in radar coordinates system; '
             'and L2: Processed instrument data in geocoded coordinates system'),
        Meta('look_direction', 'Right', 'Look direction can be left or right'),
        Meta('instrument_name', 'C-SAR', 'Instrument name'),
        Meta('orbit_pass_direction', burst.orbit_direction,
             'Orbit direction can be ascending or descending'),
        Meta('radar_band', 'C', 'Radar band'),
        Meta('zero_doppler_start_time', burst.sensing_start.strftime(TIME_STR_FMT),
             'Azimuth start time of product'),
        Meta('zero_doppler_end_time', burst.sensing_stop.strftime(TIME_STR_FMT),
            'Azimuth stop time of product'),
        Meta('is_geocoded', 'True', 'Boolean indicating if product is in radar geometry or geocoded'),
        Meta('processing_center', 'Jet Propulsion Laboratory', 'Name of the processing center that produced the product')
        ]
    id_group = dst_group.require_group('identification')
    for meta_item in id_meta_items:
        add_dataset_and_attrs(id_group, meta_item)


def metadata_to_h5group(parent_group, burst, cfg, save_noise_and_cal=True,
                        save_processing_parameters=True,
                        eap_correction_applied='None'):
    '''
    Write burst metadata to HDF5

    Parameter:
    ---------
    parent_group: h5py Group
        HDF5 group Meta data will be written to
    burst: Sentinel1BurstSlc
        Burst whose metadata is to written to HDF5
    cfg: types.SimpleNamespace
        SimpleNamespace containing run configuration
    save_noise_and_cal: bool
        If true, to save noise and calibration metadata in metadata
    save_processing_parameters: bool
        If true, to save processing parameters in metadata
    '''
    if 'metadata' in parent_group:
        del parent_group['metadata']

    # create metadata group to write datasets to
    meta_group = parent_group.require_group('metadata')

    # orbit items
    if 'orbit' in meta_group:
        del meta_group['orbit']
    orbit_group = meta_group.require_group('orbit')

    # Get orbit type
    orbit_file_path = os.path.basename(cfg.orbit_path[0])

    if 'RESORB' in orbit_file_path:
        orbit_type = 'RESORB'
    elif 'POEORB' in orbit_file_path:
        orbit_type = 'POEORB'
    else:
        err_str = f'{cfg.orbit_path[0]} is not a valid RESORB/POERB file'
        raise ValueError(err_str)

    save_orbit(burst.orbit, burst.orbit_direction,
               orbit_type, orbit_group)

    # create metadata group to write datasets to
    processing_group = meta_group.require_group('processing_information')

    # write out calibration metadata, if present
    if burst.burst_calibration is not None and save_noise_and_cal:
        cal = burst.burst_calibration
        cal_items = [
            Meta('azimuth_time', cal.azimuth_time.strftime(TIME_STR_FMT),
                 'Start time', {'format': 'YYYY-MM-DD HH:MM:SS.6f'}),
            Meta('beta_naught', cal.beta_naught, 'beta_naught')
        ]
        cal_group = meta_group.require_group('calibration_information')
        for meta_item in cal_items:
            add_dataset_and_attrs(cal_group, meta_item)

    # write out noise metadata, if present
    if burst.burst_noise is not None and save_noise_and_cal:
        noise = burst.burst_noise
        noise_items = [
            Meta('range_azimuth_time',
                 noise.range_azimuth_time.strftime(TIME_STR_FMT),
                 'Start time', {'format': 'YYYY-MM-DD HH:MM:SS.6f'})
        ]
        noise_group = meta_group.require_group('noise_information')
        for meta_item in noise_items:
            add_dataset_and_attrs(noise_group, meta_item)

    # runconfig yaml text
    processing_group['runconfig'] = cfg.yaml_string
    processing_group['runconfig'].attrs['description'] = np.string_('Run configuration file used to generate the CSLC-S1 product')

    # input items
    orbit_files = [os.path.basename(f) for f in cfg.orbit_path]
    input_items = [
        Meta('l1_slc_files', burst.safe_filename,
             'List of input L1 RSLC files used for processing'),
        Meta('orbit_files', orbit_files, 'List of input orbit files used for processing'),
        Meta('calibration_files', burst.burst_calibration.basename_cads,
             'List of input calibration files used for processing'),
        Meta('noise_files', burst.burst_noise.basename_nads,
             'List of input noise files used for processing'),
        Meta('dem_source',
             os.path.basename(cfg.groups.dynamic_ancillary_file_group.dem_description),
             'Description of the DEM used for processing'),
    ]
    input_group = processing_group.require_group('inputs')
    for meta_item in input_items:
        add_dataset_and_attrs(input_group, meta_item)

    vrt_items = [
        Meta('tiff_path', burst.tiff_path,
             'Path to measurement tiff file inside the SAFE file'),
        Meta('burst_index', burst.i_burst,
             'Burst index relative other bursts in swath'),
        Meta('first_valid_sample', burst.first_valid_sample,
             'First valid sample for burst in measurement tiff'),
        Meta('last_valid_sample', burst.last_valid_sample,
             'Last valid sample for burst in measurement tiff'),
        Meta('first_valid_line', burst.first_valid_line,
             'First valid line for burst in measurement tiff'),
        Meta('last_valid_line', burst.last_valid_line,
             'Last valid line for burst in measurement tiff')
    ]
    vrt_group = input_group.require_group('burst_location_parameters')
    for meta_item in vrt_items:
        add_dataset_and_attrs(vrt_group, meta_item)

    # burst items
    burst_meta_items = [
        Meta('ipf_version', str(burst.ipf_version),
             'ESA Instrument Processing Facility software version'),
        Meta('sensing_start', burst.sensing_start.strftime(TIME_STR_FMT),
             'Sensing start time of the burst',
             {'format': 'YYYY-MM-DD HH:MM:SS.6f'}),
        Meta('radar_center_frequency', burst.radar_center_frequency,
             'Radar center frequency', {'units':'Hertz'}),
        Meta('wavelength', burst.wavelength,
             'Wavelength of the transmitted signal', {'units':'meters'}),
        Meta('azimuth_steering_rate', burst.azimuth_steer_rate,
             'Azimuth steering rate of IW and EW modes',
             {'units':'degrees/second'}),
        Meta('azimuth_time_interval', burst.azimuth_time_interval,
             'Time spacing between azimuth lines of the burst',
             {'units':'seconds'}),
        Meta('slant_range_time', burst.slant_range_time,
             'two-way slant range time of Doppler centroid frequency estimate',
             {'units':'seconds'}),
        Meta('starting_range', burst.starting_range,
             'Slant range of the first sample of the input burst',
             {'units':'meters'}),
        Meta('sensing_stop', burst.sensing_stop.strftime(TIME_STR_FMT),
             'Sensing stop time of the burst',
             {'format': 'YYYY-MM-DD HH:MM:SS.6f'}),
        Meta('iw2_mid_range', burst.iw2_mid_range,
             'Slant range of the middle of the IW2 swath',
             {'units':'meters'}),
        Meta('range_sampling_rate', burst.range_sampling_rate,
             'Sampling rate of slant range in the input burst SLC',
             {'units':'Hertz'}),
        Meta('range_pixel_spacing', burst.range_pixel_spacing,
             'Pixel spacing between slant range samples in the input burst SLC',
             {'units':'meters'}),
        Meta('shape', burst.shape, 'Shape (length, width) of the burst in radar coordinates',
             {'units':'pixels'}),
        Meta('range_bandwidth', burst.range_bandwidth,
             'Slant range bandwidth of the signal', {'units':'Hertz'}),
        Meta('polarization', burst.polarization, 'Polarization of the burst'),
        Meta('platform_id', burst.platform_id,
             'Sensor platform identification string (e.g., S1A or S1B)'),
        Meta('center', [xy[0] for xy in burst.center.coords.xy],
             'Longitude, latitude center of burst', {'units':'degrees'}),
        # window parameters
        Meta('range_window_type', burst.range_window_type,
             'Name of the weighting window type used during processing'),
        Meta('range_window_coefficient', burst.range_window_coefficient,
             'Value of the weighting window coefficient used during processing'),
        Meta('rank', burst.rank,
             "The number of Pulse Repetition Intervals (PRI) between transmitted pulse and return echo"),
        Meta('prf_raw_data', burst.prf_raw_data,
             'Pulse repetition frequency (PRF) of the raw data',
             {'units':'Hertz'}),
        Meta('range_chirp_rate', burst.range_chirp_rate,
             'Range chirp rate', {'units':'Hertz'})
    ]
    burst_meta_group = processing_group.require_group('input_burst_metadata')
    for meta_item in burst_meta_items:
        add_dataset_and_attrs(burst_meta_group, meta_item)

    # Add parameters group in processing information
    if save_processing_parameters:
        dry_tropo_corr_enabled = \
            True if (cfg.weather_model_file is not None) and \
            ('dry' in cfg.tropo_params.delay_type) else False
        wet_tropo_corr_enabled = \
            True if (cfg.weather_model_file is not None) and \
            ('wet' in cfg.tropo_params.delay_type) else False
        tec_corr_enabled = True if cfg.tec_file is not None else False
        who_applied_eap_correction = 'OPERA' if eap_correction_applied else 'ESA'
        par_meta_items = [
            Meta('ellipsoidal_flattening_applied',
                 bool(cfg.geocoding_params.flatten),
                 "If True, CSLC-S1 phase has been flattened with respect to a zero height ellipsoid"),
            Meta('elevation_antenna_pattern_correction_applied',
                 who_applied_eap_correction,
                 ("Elevation antenna pattern correction. "
                  "OPERA: correction applied by s1-reader and COMPASS. "
                  "ESA: correction was applied by ESA. "
                  "None: when the correction was not applied.")),
            Meta('topographic_flattening_applied',
                 bool(cfg.geocoding_params.flatten),
                 "If True, CSLC-S1 phase has been flattened with respect to topographic height using a DEM"),
            Meta('bistatic_delay_applied',
                 bool(cfg.lut_params.enabled),
                 "If True, bistatic delay timing correction has been applied"),
            Meta('azimuth_fm_rate_applied',
                 bool(cfg.lut_params.enabled),
                 "If True, azimuth FM-rate mismatch timing correction has been applied"),
            Meta('geometry_doppler_applied',
                 bool(cfg.lut_params.enabled),
                 "If True, geometry steering doppler timing correction has been applied"),
            Meta('los_solid_earth_tides_applied', bool(cfg.lut_params.enabled),
                 "If True, solid Earth tides correction has been applied in slant range direction"),
            Meta('azimuth_solid_earth_tides_applied', False,
                 "If True, solid Earth tides correction has been applied in azimuth direction"),
            Meta('static_troposphere_applied',
                 bool(cfg.lut_params.enabled),
                 "If True, troposphere correction based on a static model has been applied"),
            Meta('ionosphere_tec_applied', tec_corr_enabled,
                 "If True, ionosphere correction based on TEC data has been applied"),
            Meta('dry_troposphere_weather_model_applied',
                 dry_tropo_corr_enabled,
                 "If True, dry troposphere correction based on weather model has been applied"),
            Meta('wet_troposphere_weather_model_applied',
                 wet_tropo_corr_enabled,
                 "If True, wet troposphere correction based on weather model has been applied")
        ]
        par_meta_group = processing_group.require_group('parameters')
        for meta_item in par_meta_items:
            add_dataset_and_attrs(par_meta_group, meta_item)

    def poly1d_to_h5(group, poly1d_name, poly1d):
        '''Write isce3.core.Poly1d properties to hdf5
        Parameters
        ----------
        group: h5py.Group
            h5py Group to store poly1d parameters in
        poly1d_name: str
            Name of Poly1d whose parameters are to be stored
        poly1d: isce3.core.Poly1d
            Poly1d ojbect whose parameters are to be stored
        '''
        poly1d_items = [
            Meta('order', poly1d.order, 'order of the polynomial'),
            Meta('mean', poly1d.mean, 'mean of the polynomial'),
            Meta('std', poly1d.std, 'standard deviation of the polynomial'),
            Meta('coeffs', poly1d.coeffs, 'coefficients of the polynomial'),
        ]
        poly1d_group = group.require_group(poly1d_name)
        for meta_item in poly1d_items:
            add_dataset_and_attrs(poly1d_group, meta_item)

    poly1d_to_h5(burst_meta_group, 'azimuth_fm_rate', burst.azimuth_fm_rate)
    poly1d_to_h5(burst_meta_group, 'doppler', burst.doppler.poly1d)


def algorithm_metadata_to_h5group(parent_group, is_static_layers=False):
    '''
    Write algorithm information to HDF5

    Parameter:
    ---------
    parent_group: h5py Group
        HDF5 group Meta data will be written to
    is_static_layers: bool
        True if writing algorithm metadata for static layer product
    '''
    # common algorithm items
    algorithm_items = [
        Meta('dem_interpolation', 'biquintic', 'DEM interpolation method'),
        Meta('float_data_geocoding_interpolator', 'biquintic interpolation',
             'Floating-point data geocoding interpolation method'),
        Meta('ISCE3_version', isce3.__version__,
             'ISCE3 version used for processing'),
        Meta('s1_reader_version', s1reader.__version__,
             'S1 reader version used for processing'),
        Meta('COMPASS_version', compass.__version__,
             'COMPASS (CSLC-S1 processor) version used for processing')
    ]
    if is_static_layers:
        algorithm_items.extend([
            Meta('uint_data_geocoding_interpolator',
                 'nearest neighbor interpolation',
                 'Unsigned int geocoding interpolation method'),
            Meta('topography_algorithm', 'isce3.geometry.topo',
                 'Topography generation algorithm')
        ])
    if not is_static_layers:
        algorithm_items.append(
            Meta('complex_data_geocoding_interpolator', 'sinc interpolation',
                 'Complex data geocoding interpolation method'),
        )
    algorithm_group = \
        parent_group.require_group('metadata/processing_information/algorithms')
    for meta_item in algorithm_items:
        add_dataset_and_attrs(algorithm_group, meta_item)


def corrections_to_h5group(parent_group, burst, cfg, rg_lut, az_lut,
                           scratch_path, weather_model_path=None,
                           delay_type='dry'):
    '''
    Write azimuth, slant range, and EAP (if needed) correction LUT2ds to HDF5

    Parameter:
    ---------
    parent_group: h5py Group
        HDF5 group where correction data will be written to
    burst: Sentinel1BurstSlc
        Burst containing corrections
    cfg: types.SimpleNamespace
        SimpleNamespace containing run configuration
    rg_lut: isce3.core.LUT2d()
        LUT2d along slant direction
    az_lut: isce3.core.LUT2d()
        LUT2d along azimuth direction
    scratch_path: str
        Path to the scratch directory
    weather_model_path: str
        Path to troposphere weather model in NetCDF4 format.
        This is the only format supported by RAiDER. If None,
        no weather model-based troposphere correction is applied
        (default: None).
    delay_type: str
        Type of troposphere delay. Any between 'dry', or 'wet', or
        'wet_dry' for the sum of wet and dry troposphere delays.
    '''

    # If enabled, save the correction LUTs
    if not cfg.lut_params.enabled:
        return

    # Open GDAL dataset to fetch corrections
    ds = gdal.Open(f'{scratch_path}/corrections/corrections',
                   gdal.GA_ReadOnly)
    correction_group = parent_group.require_group('timing_corrections')

    # create slant range and azimuth vectors shared by the LUTs
    x_end = rg_lut.x_start + rg_lut.width * rg_lut.x_spacing
    slant_range = np.linspace(rg_lut.x_start, x_end,
                              rg_lut.width, dtype=np.float64)
    y_end = az_lut.y_start + az_lut.length * az_lut.y_spacing
    azimuth = np.linspace(az_lut.y_start, y_end,
                          az_lut.length, dtype=np.float64)

    # correction LUTs axis and doppler correction LUTs
    desc = 'correction as a function of slant range and azimuth time'
    correction_items = [
        Meta('slant_range', slant_range, 'slant range of LUT data',
            {'units': 'meters'}),
        Meta('slant_range_spacing', rg_lut.x_spacing,
             'spacing of slant range of LUT data', {'units': 'meters'}),
        Meta('zero_doppler_time', azimuth, 'azimuth time of LUT data',
             {'units': 'seconds'}),
        Meta('zero_doppler_time_spacing', rg_lut.y_spacing,
             'spacing of azimuth time of LUT data', {'units': 'seconds'}),
        Meta('bistatic_delay', ds.GetRasterBand(2).ReadAsArray(),
             f'bistatic delay (azimuth) {desc}', {'units': 'seconds'}),
        Meta('geometry_steering_doppler', ds.GetRasterBand(1).ReadAsArray(),
             f'geometry steering doppler (range) {desc}',
             {'units': 'meters'}),
        Meta('azimuth_fm_rate_mismatch', ds.GetRasterBand(3).ReadAsArray(),
             f'azimuth FM rate mismatch mitigation (azimuth) {desc}',
             {'units': 'seconds'}),
        Meta('los_solid_earth_tides', ds.GetRasterBand(4).ReadAsArray(),
             f'Solid Earth tides (range) {desc}',
             {'units': 'meters'}),
        Meta('azimuth_solid_earth_tides', ds.GetRasterBand(5).ReadAsArray(),
             f'Solid Earth tides (azimuth) {desc}',
             {'units': 'seconds'}),
        Meta('los_ionospheric_delay', ds.GetRasterBand(6).ReadAsArray(),
             f'Ionospheric delay (range) {desc}',
             {'units': 'meters'}),
    ]
    if weather_model_path is not None:
        if 'wet' in delay_type:
            correction_items.append(Meta('wet_los_troposphere_delay',
                                         ds.GetRasterBand(7).ReadAsArray(),
                                         f'Wet LOS troposphere delay {desc}',
                                         {'units': 'meters'}))
        if 'dry' in delay_type:
            correction_items.append(Meta('dry_los_troposphere_delay',
                                         ds.GetRasterBand(8).ReadAsArray(),
                                         f'Dry LOS troposphere delay {desc}',
                                         {'units': 'meters'}))

    for meta_item in correction_items:
        add_dataset_and_attrs(correction_group, meta_item)

    # Extended FM rate and doppler centroid polynomial coefficients for azimuth
    # FM rate mismatch mitigation
    extended_coeffs = burst.extended_coeffs
    fm_rate_aztime_vec = [t.strftime(TIME_STR_FMT)
                          for t in extended_coeffs.fm_rate_aztime_vec]
    dc_aztime_vec = [t.strftime(TIME_STR_FMT)
                     for t in extended_coeffs.dc_aztime_vec]
    extended_coeffs_items = [
        Meta('fm_rate_azimuth_time', fm_rate_aztime_vec,
             'Azimuth time for FM rate coefficient data',
             {'format': 'YYYY-MM-DD HH:MM:SS.6f'}),
        Meta('fm_rate_slant_range_time', extended_coeffs.fm_rate_tau0_vec,
             'Slant range time for FM rate coefficient data',
             {'units':'seconds'}),
        Meta('fm_rate_coefficients', extended_coeffs.fm_rate_coeff_arr,
             'FM rate coefficient data'),
        Meta('doppler_centroid_azimuth_time', dc_aztime_vec,
             'Azimuth time for doppler centroid coefficient data',
             {'format': 'YYYY-MM-DD HH:MM:SS.6f'}),
        Meta('doppler_centroid_slant_range_time', extended_coeffs.dc_tau0_vec,
             'Slant range time for doppler centroid coefficient data',
             {'units':'seconds'}),
        Meta('doppler_centroid_coefficients', extended_coeffs.dc_coeff_arr,
             'Doppler centroid coefficient data')
    ]
    extended_coeffs_group = correction_group.require_group('extended_coefficients')
    for meta_item in extended_coeffs_items:
        add_dataset_and_attrs(extended_coeffs_group, meta_item)


def get_cslc_geotransform(filename: str, pol: str = "VV"):
    '''
    Extract and return geotransform of a geocoded CSLC raster in an HDFg

    Parameters
    ----------
    filename: str
        Path the CSLC HDF5
    pol: str
        Polarization of geocoded raster whose boundary is to be computed

    Returns
    -------
    list
        Geotransform of the geocoded raster
    '''
    gdal_str = f'NETCDF:{filename}:/{DATA_PATH}/{pol}'
    return gdal.Info(gdal_str, format='json')['geoTransform']


def get_georaster_bounds(filename: str, pol: str = 'VV'):
    '''
    Compute CSLC raster boundary of a given polarization

    Parameters
    ----------
    filename: str
        Path the CSLC HDF5
    pol: str
        Polarization of geocoded raster whose boundary is to be computed

    Returns
    -------
    tuple
        WGS84 coordinates of the geocoded raster boundary given as min_x,
        max_x, min_y, max_y
    '''
    nfo = gdal.Info(f'NETCDF:{filename}:/{DATA_PATH}/{pol}', format='json')

    # set extreme initial values for min/max x/y
    min_x = 999999
    max_x = -999999
    min_y = 999999
    max_y = -999999

    # extract wgs84 extent and find min/max x/y
    wgs84_coords = nfo['wgs84Extent']['coordinates'][0]
    for x, y in wgs84_coords:
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    return (min_x, max_x, min_y, max_y)

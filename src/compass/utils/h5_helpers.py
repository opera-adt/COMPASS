'''
Collection of functions to help write HDF5 datasets and metadata
'''

from dataclasses import dataclass, field
import os

import isce3
import numpy as np
from osgeo import osr
import s1reader
from s1reader.s1_reader import is_eap_correction_necessary

import compass
from compass.utils.lut import compute_geocoding_correction_luts
from compass.utils.raster_polygon import get_boundary_polygon


TIME_STR_FMT = '%Y-%m-%d %H:%M:%S.%f'


def init_geocoded_dataset(grid_group, dataset_name, geo_grid, dtype,
                          description):
    '''
    Create and allocate dataset for isce.geocode.geocode_slc to write to that
    is NC compliant

    grid_group: h5py.Group
        h5py group where geocoded dataset will be created in
    dataset_name: str
        Name of dataset to be created
    geo_grid: isce3.product.GeoGridParameters
        Geogrid of output
    dtype: str
        Data type of dataset to be geocoded
    description: str
        Description of dataset to be geocoded
    '''
    shape = (geo_grid.length, geo_grid.width)
    cslc_ds = grid_group.require_dataset(dataset_name, dtype=dtype,
                                         shape=shape)

    cslc_ds.attrs['description'] = description

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
    x_ds = grid_group.require_dataset('x_coordinates', dtype='float64',
                                      data=x_vect, shape=x_vect.shape)
    y_ds = grid_group.require_dataset('y_coordinates', dtype='float64',
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

    grid_meta_items = [
        Meta('x_spacing', geo_grid.spacing_x,
             'Spacing of geo grid in x-axis.'),
        Meta('y_spacing', geo_grid.spacing_y,
             'Spacing of geo grid in y-axis.')
    ]
    for meta_item in grid_meta_items:
        add_dataset_and_attrs(grid_group, meta_item)

    # Set up osr for wkt
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(geo_grid.epsg)

    #Create a new single int dataset for projections
    projection_ds = grid_group.require_dataset('projection', (), dtype='i')
    projection_ds[()] = geo_grid.epsg

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
        x_ds.attrs['standard_name'] = np.string_("projection_x_coordinate")
        x_ds.attrs['long_name'] = np.string_("x coordinate of projection")
        x_ds.attrs['units'] = np.string_("m")

        y_ds.attrs['standard_name'] = np.string_("projection_y_coordinate")
        y_ds.attrs['long_name'] = np.string_("y coordinate of projection")
        y_ds.attrs['units'] = np.string_("m")

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
    group[meta_item.name] = val

    # Write data and attributes
    val_ds = group[meta_item.name]
    desc = _as_np_string_if_needed(meta_item.description)
    val_ds.attrs['description'] = desc
    for key, val in meta_item.attr_dict.items():
        val_ds.attrs[key] = _as_np_string_if_needed(val)


def save_orbit(orbit, orbit_direction, orbit_group):
    '''
    Write burst to HDF5

    Parameter
    ---------
    orbit: isce3.core.Orbit
        ISCE3 orbit object
    orbit_group: h5py.Group
        HDF5 group where orbit parameters will be written
    '''
    ref_epoch = orbit.reference_epoch.isoformat().replace('T', ' ')
    orbit_items = [
        Meta('ref_epoch', ref_epoch, 'Reference epoch of the state vectors',
             {'format': 'YYYY-MM-DD HH:MM:SS.6f'}),
        Meta('time', np.linspace(orbit.time.first,
                                 orbit.time.last,
                                 orbit.time.size),
             'Time of the orbit state vectors relative to the reference epoch',
             {'units': 'seconds'}),
        Meta('orbit_direction', orbit_direction,
             'Direction of sensor orbit ephermerides (e.g., ascending, descending)')
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
                                           data=np.string_("POE"))
    orbit_ds.attrs["description"] = np.string_("PrOE (or) NOE (or) MOE (or) POE"
                                        " (or) Custom")


def identity_to_h5group(dst_group, burst, dataset_path):
    '''
    Write burst metadata to HDF5

    Parameter:
    ---------
    dst_group: h5py Group
        HDF5 group metadata will be written to
    burst: Sentinel1BurstSlc
        Burst whose metadata is to written to HDF5
    dataset_path: str
        Path to CSLC data in HDF5
    '''
    # identification datasets
    # TODO will need some changes to accommodate dateline
    dataset_path_template = f'HDF5:%FILE_PATH%:/{dataset_path}'
    geo_boundary = get_boundary_polygon(dst_group.file.filename, np.nan,
                                        dataset_path_template)
    id_meta_items = [
        Meta('product_version', '?', 'CSLC product version'),
        Meta('absolute_orbit_number', burst.abs_orbit_number, 'Absolute orbit number'),
        # NOTE: The field below does not exist on opera_rtc.xml
        # 'relativeOrbitNumber':
        #   [int(burst.burst_id[1:4]), 'Relative orbit number'],
        Meta('track_number', burst.burst_id.track_number, 'Track number'),
        Meta('burst_ID', str(burst.burst_id), 'Burst identification (burst ID)'),
        Meta('bounding_polygon', str(geo_boundary),
             'OGR compatible WKT representation of bounding polygon of the image'),
        Meta('mission_ID', burst.platform_id, 'Mission identifier'),
        Meta('product_type', 'CSLC-S1', 'Product type'),
        # NOTE: in NISAR, the value has to be in UPPERCASE or lowercase?
        Meta('look_direction', 'Right', 'Look direction can be left or right'),
        Meta('orbit_pass_direction', burst.orbit_direction,
             'Orbit direction can be ascending or descending'),
        # NOTE: using the same date format as `s1_reader.as_datetime()`
        Meta('zero_doppler_start_time', burst.sensing_start.strftime(TIME_STR_FMT),
             'Azimuth start time of product'),
        Meta('zero_doppler_end_time', burst.sensing_stop.strftime(TIME_STR_FMT),
            'Azimuth stop time of product'),
        Meta('list_of_frequencies', ['A'],
             'List of frequency layers available in the product'),  # T)C
        Meta('is_geocoded', True, 'Flag to indicate radar geometry or geocoded product'),
        Meta('is_urgent_observation', False,
             'List of booleans indicating if datatakes are nominal or urgent'),
        Meta('diagnostic_mode_flag', False,
             'Indicates if the radar mode is a diagnostic mode or not: True or False'),
        # missing:
        # 'processingType'
        # 'productVersion'
        # 'frameNumber':  # TBD
        # 'productVersion': # Defined by RTC SAS
        # 'plannedDatatakeId':
        # 'plannedObservationId':
        ]
    id_group = dst_group.require_group('identification')
    for meta_item in id_meta_items:
        add_dataset_and_attrs(id_group, meta_item)


def metadata_to_h5group(parent_group, burst, cfg):
    '''
    Write burst metadata to HDF5

    Parameter:
    ---------
    parent_group: h5py Group
        HDF5 group Meta data will be written to
    burst: Sentinel1BurstSlc
        Burst whose metadata is to written to HDF5
    geogrid: isce3.product.GeoGridParameters
        Geo grid object defining the geocoded area
    cfg: types.SimpleNamespace
        SimpleNamespace containing run configuration
    '''
    # create metadata group to write datasets to
    meta_group = parent_group.require_group('metadata')

    # orbit items
    if 'orbit' in meta_group:
        del meta_group['orbit']
    orbit_group = meta_group.require_group('orbit')
    save_orbit(burst.orbit, burst.orbit_direction, orbit_group)

    # create metadata group to write datasets to
    processing_group = meta_group.require_group('processing_information')

    # input items
    l1_slc_files = [os.path.basename(f) for f in cfg.safe_files]
    orbit_files = [os.path.basename(f) for f in cfg.orbit_path]
    input_items = [
        Meta('l1_slc_files', l1_slc_files, 'Input L1 RSLC files used'),
        Meta('orbit_files', orbit_files, 'List of input orbit files used'),
        Meta('calibration_file', burst.burst_calibration.basename_cads,
             'Input calibration file used'),
        Meta('noise_file', burst.burst_noise.basename_nads,
             'Input noise file used'),
        Meta('dem_source', os.path.basename(cfg.dem), 'sorce DEM file')
    ]
    input_group = processing_group.require_group('inputs')
    for meta_item in input_items:
        add_dataset_and_attrs(input_group, meta_item)

    # algorithm items
    algorithm_items = [
        Meta('dem_interpolation', 'biquintic', 'DEM interpolation method'),
        Meta('geocoding', 'sinc interpolation', 'Geocoding algorithm'),
        Meta('ISCE_version', isce3.__version__,
             'ISCE version used for processing'),
        Meta('s1Reader_version', s1reader.__version__,
             'S1-Reader version used for processing'),
        Meta('COMPASS_version', compass.__version__,
             'COMPASS version used for processing')
    ]
    algorithm_group = processing_group.require_group('algorithms')
    for meta_item in algorithm_items:
        add_dataset_and_attrs(algorithm_group, meta_item)

    # burst items
    burst_meta_items = [
        Meta('sensing_start', burst.sensing_start.strftime(TIME_STR_FMT),
             'Sensing start time of the burst',
             {'format': 'YYYY-MM-DD HH:MM:SS.6f'}),
        Meta('sensing_stop', burst.sensing_stop.strftime(TIME_STR_FMT),
             'Sensing stop time of the burst',
             {'format': 'YYYY-MM-DD HH:MM:SS.6f'}),
        Meta('radar_center_frequency', burst.radar_center_frequency,
             'Radar center frequency', {'units':'Hz'}),
        Meta('wavelength', burst.wavelength,
             'Wavelength of the transmitted signal', {'units':'meters'}),
        Meta('azimuth_steer_rate', burst.azimuth_steer_rate,
             'Azimuth steering rate of IW and EW modes',
             {'units':'degrees/second'}),
        Meta('azimuth_time_interval', burst.azimuth_time_interval,
             'Time spacing between azimuth lines of the burst',
             {'units':'seconds'}),
        Meta('starting_range', burst.starting_range,
             'Slant range of the first sample of the input burst',
             {'units':'meters'}),
        Meta('slant_range_time', burst.slant_range_time,
             'two-way slant range time of Doppler centroid frequency estimate',
             {'units':'seconds'}),
        Meta('range_pixel_spacing', burst.range_pixel_spacing,
             'Pixel spacing between slant range samples in the input burst SLC',
             {'units':'meters'}),
        Meta('range_bandwidth', burst.range_bandwidth,
             'Slant range bandwidth of the signal', {'units':'Hz'}),
        Meta('polarization', burst.polarization, 'Polarization of the burst'),
        Meta('platform_id', burst.platform_id,
             'Sensor platform identification string (e.g., S1A or S1B)'),
        # window parameters
        Meta('range_window_type', burst.range_window_type,
             'name of the weighting window type used during processing'),
        Meta('range_window_coefficient', burst.range_window_coefficient,
             'value of the weighting window coefficient used during processing')
    ]
    burst_meta_group = processing_group.require_group('s1_burst_metadata')
    for meta_item in burst_meta_items:
        add_dataset_and_attrs(burst_meta_group, meta_item)

    # runconfig yaml text
    meta_group['runconfig'] = cfg.yaml_string


def corrections_to_h5group(parent_group, burst, cfg):
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
    '''
    correction_group = parent_group.require_group('corrections')

    # Get range and azimuth LUTs
    geometrical_steering_doppler, bistatic_delay_lut, az_fm_rate = \
        compute_geocoding_correction_luts(burst,
                                          dem_path=cfg.dem,
                                          rg_step=cfg.lut_params.range_spacing,
                                          az_step=cfg.lut_params.azimuth_spacing)

    # create linspace for axises shared by both LUTs
    x_end = bistatic_delay_lut.x_start + bistatic_delay_lut.width * bistatic_delay_lut.x_spacing
    slant_range = np.linspace(bistatic_delay_lut.x_start, x_end,
                              bistatic_delay_lut.width, dtype=np.float64)
    y_end = bistatic_delay_lut.y_start + bistatic_delay_lut.length * bistatic_delay_lut.y_spacing
    azimuth = np.linspace(bistatic_delay_lut.y_start, y_end,
                          bistatic_delay_lut.length, dtype=np.float64)

    # correction LUTs axis and doppler correction LUTs
    desc = ' correction as a function of slant range and azimuth time'
    correction_items = [
        Meta('slant_range', slant_range, 'slant range of LUT data',
             {'units': 'meters'}),
        Meta('zero_doppler_time', azimuth, 'azimuth time of LUT data',
             {'units': 'seconds'}),
        Meta('bistatic_delay', bistatic_delay_lut.data,
             f'bistatic delay (azimuth) {desc}', {'units': 'seconds'}),
        Meta('geometry_steering_doppler', geometrical_steering_doppler.data,
             f'geometry steering doppler (range) {desc}',
             {'units': 'meters'}),
        Meta('azimuth_fm_rate_mismatch_mitigation', az_fm_rate.data,
             f'azimuth FM rate mismatch mitigation (azimuth) {desc}',
             {'units': 'seconds'}),
    ]
    for meta_item in correction_items:
        add_dataset_and_attrs(correction_group, meta_item)

    # EAP metadata depending on IPF version
    check_eap = is_eap_correction_necessary(burst.ipf_version)
    if check_eap.phase_correction:
        eap = burst.burst_eap
        eap_items = [
            Meta('sampling_frequency', eap.freq_sampling,
                 'range sampling frequency', { 'units': 'Hz'}),
            Meta('eta_start', eap.eta_start.strftime(TIME_STR_FMT),
                 'Sensing start time', {'format': 'YYYY-MM-DD HH:MM:SS.6f'}),
            Meta('tau_0', eap.tau_0, 'slant range time', {'units': 'seconds'}),
            Meta('tau_sub', eap.tau_sub, 'slant range time',
                 {'units': 'seconds'}),
            Meta('theta_sub', eap.theta_sub, 'elevation angle',
                 {'units': 'radians'}),
            Meta('ascending_node_time',
                 eap.ascending_node_time.strftime(TIME_STR_FMT),
                 'Ascending node crossing time (ANX)',
                 {'format': 'YYYY-MM-DD HH:MM:SS.6f'})
        ]
        eap_group = correction_group.require_group('elevation_antenna_pattern')
        for meta_item in eap_items:
            add_dataset_and_attrs(eap_group, meta_item)

'''
Collection of functions to help write HDF5 datasets and metadata
'''

from dataclasses import dataclass, field
import os

import isce3
import numpy as np
from osgeo import osr, gdal
import s1reader
from s1reader.s1_burst_slc import Sentinel1BurstSlc
import shapely

import compass
from compass.utils.lut import compute_geocoding_correction_luts


TIME_STR_FMT = '%Y-%m-%d %H:%M:%S.%f'


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
        Meta('reference_epoch', ref_epoch, 'Reference epoch of the state vectors',
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


def identity_to_h5group(dst_group, burst):
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
    id_meta_items = [
        Meta('product_version', '?', 'CSLC product version'),
        Meta('absolute_orbit_number', burst.abs_orbit_number, 'Absolute orbit number'),
        Meta('track_number', burst.burst_id.track_number, 'Track number'),
        Meta('burst_id', str(burst.burst_id), 'Burst identification (burst ID)'),
        Meta('bounding_polygon', get_polygon_wkt(burst),
             'OGR compatible WKT representation of bounding polygon of the image',
             {'units':'degrees'}),
        Meta('mission_id', burst.platform_id, 'Mission identifier'),
        Meta('product_type', 'CSLC-S1', 'Product type'),
        Meta('look_direction', 'Right', 'Look direction can be left or right'),
        Meta('orbit_pass_direction', burst.orbit_direction,
             'Orbit direction can be ascending or descending'),
        Meta('zero_doppler_start_time', burst.sensing_start.strftime(TIME_STR_FMT),
             'Azimuth start time of product'),
        Meta('zero_doppler_end_time', burst.sensing_stop.strftime(TIME_STR_FMT),
            'Azimuth stop time of product'),
        Meta('is_geocoded', 'True', 'Flag to indicate radar geometry or geocoded product'),
        Meta('is_urgent_observation', 'False',
             'List of booleans indicating if datatakes are nominal or urgent'),
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
    if 'metadata' in parent_group:
        del parent_group['metadata']

    # create metadata group to write datasets to
    meta_group = parent_group.require_group('metadata')

    # orbit items
    if 'orbit' in meta_group:
        del meta_group['orbit']
    orbit_group = meta_group.require_group('orbit')
    save_orbit(burst.orbit, burst.orbit_direction, orbit_group)

    # create metadata group to write datasets to
    processing_group = meta_group.require_group('processing_information')

    # runconfig yaml text
    processing_group['runconfig'] = cfg.yaml_string

    # input items
    orbit_files = [os.path.basename(f) for f in cfg.orbit_path]
    input_items = [
        Meta('l1_slc_files', burst.safe_filename, 'Input L1 RSLC file used'),
        Meta('orbit_files', orbit_files, 'List of input orbit files used'),
        Meta('calibration_file', burst.burst_calibration.basename_cads,
             'Input calibration file used'),
        Meta('noise_file', burst.burst_noise.basename_nads,
             'Input noise file used'),
        Meta('dem_source', os.path.basename(cfg.dem), 'sorce DEM file'),
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
    vrt_group = input_group.require_group('vrt_parameters')
    for meta_item in vrt_items:
        add_dataset_and_attrs(vrt_group, meta_item)

    # algorithm items
    algorithm_items = [
        Meta('dem_interpolation', 'biquintic', 'DEM interpolation method'),
        Meta('geocoding_interpolator', 'sinc interpolation',
             'Geocoding interpolation method'),
        Meta('ISCE3_version', isce3.__version__,
             'ISCE3 version used for processing'),
        Meta('s1Reader_version', s1reader.__version__,
             'S1-Reader version used for processing'),
        Meta('COMPASS_version', compass.__version__,
             'COMPASS (CSLC-S1 processor) version used for processing')
    ]
    algorithm_group = processing_group.require_group('algorithms')
    for meta_item in algorithm_items:
        add_dataset_and_attrs(algorithm_group, meta_item)

    # burst items
    burst_meta_items = [
        Meta('ipf_version', str(burst.ipf_version),
             'Image Processing Facility software version'),
        Meta('sensing_start', burst.sensing_start.strftime(TIME_STR_FMT),
             'Sensing start time of the burst',
             {'format': 'YYYY-MM-DD HH:MM:SS.6f'}),
        Meta('radar_center_frequency', burst.radar_center_frequency,
             'Radar center frequency', {'units':'Hz'}),
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
             {'units':'Hz'}),
        Meta('range_pixel_spacing', burst.range_pixel_spacing,
             'Pixel spacing between slant range samples in the input burst SLC',
             {'units':'meters'}),
        Meta('shape', burst.shape, 'Shape (length, width) of the burst in radar coordinates',
             {'units':'pixels'}),
        Meta('range_bandwidth', burst.range_bandwidth,
             'Slant range bandwidth of the signal', {'units':'Hz'}),
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
             {'units':'Hz'}),
        Meta('range_chirp_rate', burst.range_chirp_rate,
             'Range chirp rate', {'units':'Hz'})
    ]
    burst_meta_group = processing_group.require_group('s1_burst_metadata')
    for meta_item in burst_meta_items:
        add_dataset_and_attrs(burst_meta_group, meta_item)

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


def corrections_to_h5group(parent_group, burst, cfg, rg_lut, az_lut,
                           scratch_path):
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
    '''

    # If enabled, save the correction LUTs
    if cfg.lut_params.enabled:
        # Open GDAL dataset to fetch corrections
        ds = gdal.Open(f'{scratch_path}/corrections/corrections',
                       gdal.GA_ReadOnly)
        correction_group = parent_group.require_group('corrections')


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
            Meta('zero_doppler_time_spacing',rg_lut.y_spacing,
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
        ]
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

    # write out EAP metadata, if present
    if burst.burst_eap is not None:
        eap = burst.burst_eap
        eap_items = [
            Meta('sampling_frequency', eap.freq_sampling,
                 'range sampling frequency', { 'units': 'Hz'}),
            Meta('eta_start', eap.eta_start.strftime(TIME_STR_FMT),
                 'Sensing start time', {'format': 'YYYY-MM-DD HH:MM:SS.6f'}),
            Meta('tau_0', eap.tau_0, 'slant range time of the product', {'units': 'seconds'}),
            Meta('tau_sub', eap.tau_sub, 'slant range time of AUX_CAL antenna pattern',
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

    # write out calibration metadata, if present
    if burst.burst_calibration is not None:
        cal = burst.burst_calibration
        cal_items = [
            Meta('basename', cal.basename_cads, ''),
            Meta('azimuth_time', cal.azimuth_time.strftime(TIME_STR_FMT),
                 'Start time', {'format': 'YYYY-MM-DD HH:MM:SS.6f'}),
            Meta('line', cal.line, 'line'),
            Meta('pixel', cal.pixel, 'pixel'),
            Meta('sigma_naught', cal.sigma_naught, 'sigma_naught'),
            Meta('beta_naught', cal.beta_naught, 'beta_naught'),
            Meta('gamma', cal.gamma, 'gamma'),
            Meta('dn', cal.dn, 'dn'),
        ]
        cal_group = correction_group.require_group('calibration')
        for meta_item in cal_items:
            add_dataset_and_attrs(cal_group, meta_item)

    # write out noise metadata, if present
    if burst.burst_noise is not None:
        noise = burst.burst_noise
        noise_items = [
            Meta('basename', noise.basename_nads, ''),
            Meta('range_azimuth_time',
                 noise.range_azimuth_time.strftime(TIME_STR_FMT),
                 'Start time', {'format': 'YYYY-MM-DD HH:MM:SS.6f'}),
            Meta('range_line', noise.range_line, 'Range line'),
            Meta('range_pixel', noise.range_pixel, 'Range array in pixel for LUT'),
            Meta('range_lut', noise.range_lut, 'Range noise lookup table data'),
            Meta('azimuth_first_azimuth_line', noise.azimuth_first_azimuth_line,
                 'First line of the burst in subswath. NaN if not available in annotation.'),
            Meta('azimuth_first_range_sample', noise.azimuth_first_range_sample,
                 'First range sample of the burst. NaN if not available in annotation.'),
            Meta('azimuth_last_azimuth_line', noise.azimuth_last_azimuth_line,
                 'Last line of the burst in subswatn. NaN if not available in annotation.'),
            Meta('azimuth_last_range_sample', noise.azimuth_last_range_sample,
                 'Last range of the burst. NaN if not available in annotation.'),
            Meta('azimuth_line', noise.azimuth_line, 'azimuth line index for noise LUT'),
            Meta('azimuth_lut', noise.azimuth_lut, 'azimuth noise lookup table data')
        ]
        noise_group = correction_group.require_group('noise')
        for meta_item in noise_items:
            add_dataset_and_attrs(noise_group, meta_item)

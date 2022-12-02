from datetime import datetime

import isce3
import numpy as np
from osgeo import osr

import compass
from compass.utils.raster_polygon import get_boundary_polygon

def init_geocoded_dataset(geocoded_group, dataset_name, geo_grid, dtype,
                          description):
    '''
    Create and allocate dataset for isce.geocode.geocode_slc to write to that
    is NC compliant

    geocoded_group: h5py.Group
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
    geocoded_ds = geocoded_group.require_dataset(dataset_name, dtype=dtype,
                                                 shape=shape)

    geocoded_ds.attrs['description'] = description

    # Compute x scale
    dx = geo_grid.spacing_x
    x0 = geo_grid.start_x + 0.5 * dx
    xf = x0 + (geo_grid.width - 1) * dx
    x_vect = np.linspace(x0, xf, geo_grid.width, dtype=np.float64)

    # Compute y scale
    dy = geo_grid.spacing_y
    y0 = geo_grid.start_y + 0.5 * dy
    yf = y0 + (geo_grid.length - 1) * dy
    y_vect = np.linspace(y0, yf - dy, geo_grid.length, dtype=np.float64)

    # following copied and pasted (and slightly modified) from:
    # https://github-fn.jpl.nasa.gov/isce-3/isce/wiki/CF-Conventions-and-Map-Projections
    x_ds = geocoded_group.require_dataset('x', dtype='float64', data=x_vect,
                                          shape=x_vect.shape)
    y_ds = geocoded_group.require_dataset('y', dtype='float64', data=y_vect,
                                          shape=y_vect.shape)

    # Mapping of dimension scales to datasets is not done automatically in HDF5
    # We should label appropriate arrays as scales and attach them to datasets
    # explicitly as show below.
    x_ds.make_scale()
    geocoded_ds.dims[1].attach_scale(x_ds)
    y_ds.make_scale()
    geocoded_ds.dims[0].attach_scale(y_ds)

    # Associate grid mapping with data - projection created later
    geocoded_ds.attrs['grid_mapping'] = np.string_("projection")

    # Set up osr for wkt
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(geo_grid.epsg)

    #Create a new single int dataset for projections
    projection_ds = geocoded_group.require_dataset('projection', (), dtype='i')
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


def geo_burst_metadata_to_hdf5(dst_h5, burst, geogrid, cfg):
    '''Write burst metadata to HDF5

    Parameter:
    ---------
    dst_h5: h5py File
        HDF5 file meta data will be written to
    burst: Sentinel1BurstSlc
        Burst
    '''
    time_str_fmt = 'time_str_fmt'
    metadata_group = dst_h5.require_group('metadata')

    def add_dataset_and_attrs(group, name, value, attr_dict):
        '''Write isce3.core.Poly1d properties to hdf5

        Parameters
        ----------
        group: h5py.Group
            h5py Group to store poly1d parameters in
        name: str
            Name of dataset to add
        value: object
            Value to be added
        attr_dict: dict[str: object]
            Dict with attribute name as key and some object as value
        '''
        if name in group:
            del group[name]

        group[name] = value
        val_ds = group[name]
        for key, val in attr_dict.items():
            val_ds.attrs[key] = val

    # product identification and processing information
    id_proc_group = metadata_group.require_group('identifcation_and_processing')
    add_dataset_and_attrs(id_proc_group, 'product_id', 'L2_CSLC_S1', {})
    add_dataset_and_attrs(id_proc_group, 'product_version', '?', {})
    add_dataset_and_attrs(id_proc_group, 'software_version',
                          compass.__version__,
                          {'description': 'COMPASS  version used to generate the L2_CSLC_S1 product'})
    add_dataset_and_attrs(id_proc_group, 'isce3_version',
                          isce3.__version__,
                          {'description': 'ISCE3 version used to generate the L2_CSLC_S1 product'})
    add_dataset_and_attrs(id_proc_group, 'project', 'OPERA', {})
    add_dataset_and_attrs(id_proc_group, 'product_level', '2', {})
    add_dataset_and_attrs(id_proc_group, 'product_type', 'CSLC_S1', {})
    add_dataset_and_attrs(id_proc_group, 'processing_datetime',
                          datetime.now().strftime(time_str_fmt),
                          {'description': 'L2_CSLC_S1 product processing date and time',
                           'format': 'YYYY-MM-DDD HH:MM:SS'})
    add_dataset_and_attrs(id_proc_group, 'spacecraft_name',
                          burst.platform_id,
                          {'description': 'Name of Sensor platform (e.g., S1-A/B)'})

    # burst metadata
    s1ab_group = metadata_group.require_group('s1ab_burst_metadata')
    add_dataset_and_attrs(s1ab_group, 'sensing_start',
                          burst.sensing_start.strftime('time_str_fmt.%f'),
                          {'description': 'Sensing start time of the burst',
                           'format': 'YYYY-MM-DD HH:MM:SS.6f'})
    add_dataset_and_attrs(s1ab_group, 'sensing_stop',
                          burst.sensing_stop.strftime('time_str_fmt.%f'),
                          {'description':'Sensing stop time of the burst',
                           'format': 'YYYY-MM-DD HH:MM:SS.6f'})
    add_dataset_and_attrs(s1ab_group, 'radar_center_frequency',
                          burst.radar_center_frequency,
                          {'description':'Radar center frequency',
                           'units':'Hz'})
    add_dataset_and_attrs(s1ab_group, 'wavelength', burst.wavelength,
                          {'description':'Wavelength of the transmitted signal',
                           'units':'meters'})
    add_dataset_and_attrs(s1ab_group, 'azimuth_steer_rate',
                          burst.azimuth_steer_rate,
                          {'description':'Azimuth steering rate of IW and EW modes',
                           'units':'degrees/second'})
    # TODO add input width and length
    add_dataset_and_attrs(s1ab_group, 'azimuth_time_interval',
                          burst.azimuth_time_interval,
                          {'description':'Time spacing between azimuth lines of the burst',
                           'units':'seconds'})
    add_dataset_and_attrs(s1ab_group, 'starting_range',
                          burst.starting_range,
                          {'description':'Slant range of the first sample of the input burst',
                           'units':'meters'})
    # TODO do we need this? It's not in the specs.
    # TODO add far_range?
    add_dataset_and_attrs(s1ab_group, 'slant_range_time',
                          burst.slant_range_time,
                          {'description':'two-way slant range time of Doppler centroid frequency estimate',
                           'units':'seconds'})
    add_dataset_and_attrs(s1ab_group, 'range_pixel_spacing',
                          burst.range_pixel_spacing,
                          {'description':'Pixel spacing between slant range samples in the input burst SLC',
                           'units':'meters'})
    add_dataset_and_attrs(s1ab_group, 'range_bandwidth',
                          burst.range_bandwidth,
                          {'description':'Slant range bandwidth of the signal',
                           'units':'Hz'})
    add_dataset_and_attrs(s1ab_group, 'polarization',
                          burst.polarization,
                          {'description': 'Polarization of the burst'})
    add_dataset_and_attrs(s1ab_group, 'platform_id',
                          burst.platform_id,
                          {'description': 'Sensor platform identification string (e.g., S1A or S1B)'})
    # window parameters
    add_dataset_and_attrs(s1ab_group, 'range_window_type',
                          burst.range_window_type,
                          {'description': 'name of the weighting window type used during processing'})
    add_dataset_and_attrs(s1ab_group, 'range_window_coefficient',
                          burst.range_window_coefficient,
                          {'description': 'value of the weighting window coefficient used during processing'})

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
        poly1d_group = group.require_group(poly1d_name)
        add_dataset_and_attrs(poly1d_group, 'order', poly1d.order,
                              {'description': 'order of the polynomial'})
        add_dataset_and_attrs(poly1d_group, 'mean', poly1d.mean,
                              {'description': 'mean of the polynomial'})
        add_dataset_and_attrs(poly1d_group, 'std', poly1d.std,
                              {'description': 'standard deviation of the polynomial'})
        add_dataset_and_attrs(poly1d_group, 'coeffs', poly1d.coeffs,
                              {'description': 'coefficients of the polynomial'})
    poly1d_to_h5(s1ab_group, 'azimuth_fm_rate', burst.azimuth_fm_rate)
    poly1d_to_h5(s1ab_group, 'doppler', burst.doppler.poly1d)

    # EAP metadata only written if it exists
    if burst.burst_eap is not None:
        eap_group = s1ab_group.require_group('elevation_antenna_pattern_correction')
        eap = burst.burst_eap
        add_dataset_and_attrs(eap_group, 'sampling_frequency',
                              eap.freq_sampling,
                              {'description': 'range sampling frequency',
                               'units': 'Hz'})
        add_dataset_and_attrs(eap_group, 'eta_start',
                              eap.eta_start.strftime(time_str_fmt),
                              {'description': '?',
                               'format': 'YYYY-MM-DD HH:MM:SS.6f'})
        add_dataset_and_attrs(eap_group, 'tau_0', eap.tau_0,
                              {'description': 'slant range time',
                               'units': 'seconds'})
        add_dataset_and_attrs(eap_group, 'tau_sub', eap.tau_sub,
                              {'description': 'slant range time',
                               'units': 'seconds'})
        add_dataset_and_attrs(eap_group, 'theta_sub', eap.theta_sub,
                              {'description': 'elevation angle',
                               'units': 'radians'})
        add_dataset_and_attrs(eap_group, 'azimuth_time',
                              eap.azimuth_time.strftime(time_str_fmt),
                              {'description': '?',
                               'format': 'YYYY-MM-DD HH:MM:SS.6f'})
        add_dataset_and_attrs(eap_group, 'ascending_node_time',
                              eap.ascending_node_time.strftime(time_str_fmt),
                              {'description': '?',
                               'format': 'YYYY-MM-DD HH:MM:SS.6f'})

    # save orbit
    orbit_group = metadata_group.require_group('orbit')
    ref_epoch = burst.orbit.reference_epoch.isoformat().replace('T', ' ')
    add_dataset_and_attrs(orbit_group, 'ref_epoch', ref_epoch,
                          {'description': 'Reference epoch of the state vectors',
                           'format': 'YYYY-MM-DD HH:MM:SS.6f'})
    orbit_time_obj = burst.orbit.time
    add_dataset_and_attrs(orbit_group, 'time',
                          np.linspace(orbit_time_obj.first,
                                      orbit_time_obj.last,
                                      orbit_time_obj.size),
                          {'description': 'Time of the orbit state vectors relative to the reference epoch',
                           'units': 'seconds'})
    for i_ax, axis in enumerate('xyz'):
        add_dataset_and_attrs(orbit_group, f'position_{axis}',
                              burst.orbit.position[:, i_ax],
                              {'description': f'Platform position along {axis}-direction',
                               'units': 'meters'})
        add_dataset_and_attrs(orbit_group, f'velocity_{axis}',
                              burst.orbit.velocity[:, i_ax],
                              {'description': f'Platform velocity along {axis}-direction',
                               'units': 'meters/second'})
    add_dataset_and_attrs(orbit_group, 'orbit_direction',
                          burst.orbit_direction,
                          {'description':'Direction of sensor orbit ephermerides (e.g., ascending, descending)'})

    # input params
    input_group = metadata_group.require_group('input')
    add_dataset_and_attrs(input_group, 'burst_file_path',
                          burst.tiff_path,
                          {'description': 'Path to TIFF file within the SAFE file containing the burst'})
    add_dataset_and_attrs(input_group, 'input_data_ipf_version',
                          str(burst.ipf_version),
                          {'description': 'Version of Instrument Processing Facility used to generate SAFE file'})
    add_dataset_and_attrs(input_group, 'dem_file', cfg.dem,
                          {'description': 'Path to DEM file'})

    # processing params
    processing_group = metadata_group.require_group('processing')
    add_dataset_and_attrs(processing_group, 'burst_id',
                          str(burst.burst_id),
                          {'description': 'Unique burst identification string (ESA convention)'})

    dataset_path_template = f'HDF5:%FILE_PATH%://complex_backscatter/{burst.polarization}'
    geo_boundary = get_boundary_polygon(processing_group.file.filename, np.nan,
                                        dataset_path_template)
    center = geo_boundary.centroid
    center_lon_lat = np.array([val[0] for val in center.coords.xy])
    add_dataset_and_attrs(processing_group, 'center',
                          center_lon_lat,
                          {'description': 'Burst geographical center coordinates in the projection system used for processing',
                           'units': 'meters'})
    # list of coordinate tuples (in degrees) representing burst border
    border_x, border_y = geo_boundary.exterior.coords.xy
    border_lon_lat = np.array([[lon, lat] for lon, lat in zip(border_x,
                                                              border_y)])
    add_dataset_and_attrs(processing_group, 'border', border_lon_lat,
                          {'description': 'X- and Y- coordinates of the polygon including valid L2_CSLC_S1 data',
                           'units': 'meters'})
    add_dataset_and_attrs(processing_group, 'i_burst', burst.i_burst,
                          {'description': 'Index of the burst of interest relative to other bursts in the S1-A/B SAFE file'})
    add_dataset_and_attrs(processing_group, 'start_x', geogrid.start_x,
                          {'description': 'X-coordinate of the L2_CSLC_S1 starting point in the coordinate system selected for processing',
                           'units': 'meters'})
    add_dataset_and_attrs(processing_group, 'start_y', geogrid.start_y,
                          {'description': 'Y-coordinate of the L2_CSLC_S1 starting point in the coordinate system selected for processing',
                           'units': 'meters'})
    add_dataset_and_attrs(processing_group, 'x_posting',
                          geogrid.spacing_x,
                          {'description': 'Spacing between product pixels along the X-direction ',
                           'units': 'meters'})
    add_dataset_and_attrs(processing_group, 'y_posting',
                          geogrid.spacing_y,
                              {'description': 'Spacing between product pixels along the Y-direction ',
                               'units': 'meters'})
    add_dataset_and_attrs(processing_group, 'width', geogrid.width,
                          {'description': 'Number of samples in the L2_CSLC_S1 product'})
    add_dataset_and_attrs(processing_group, 'length', geogrid.length,
                          {'description': 'Number of lines in the L2_CSLC_S1 product'})
    add_dataset_and_attrs(processing_group, 'epsg',
                          geogrid.epsg,
                          {'description': 'EPSG code identifying the coordinate system used for processing'})
    add_dataset_and_attrs(processing_group, 'no_data_value', 'NaN',
                          {'description': 'Value used when no data present'})

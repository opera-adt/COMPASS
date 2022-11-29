from dataclasses import dataclass
from datetime import datetime
import json
from types import SimpleNamespace

import isce3
from isce3.core import LUT2d, Poly1d, Orbit
from isce3.product import GeoGridParameters
import numpy as np
from ruamel.yaml import YAML
from shapely.geometry import Point, Polygon

import compass
from compass.utils.geo_runconfig import GeoRunConfig
from compass.utils.raster_polygon import get_boundary_polygon
from compass.utils.wrap_namespace import wrap_namespace, unwrap_to_dict

def _poly1d_from_dict(poly1d_dict) -> Poly1d:
    return Poly1d(poly1d_dict['coeffs'], poly1d_dict['mean'],
                  poly1d_dict['std'])


def _lut2d_from_dict(lut2d_dict) -> LUT2d:
    lut2d_shape = (lut2d_dict['length'], lut2d_dict['width'])
    lut2d_data = np.array(lut2d_dict['data']).reshape(lut2d_shape)
    return LUT2d(lut2d_dict['x_start'], lut2d_dict['y_start'],
                 lut2d_dict['x_spacing'], lut2d_dict['y_spacing'],
                 lut2d_data)


def _orbit_from_dict(orbit_dict) -> Orbit:
    ref_epoch = isce3.core.DateTime(orbit_dict['ref_epoch'])

    # build state vector
    dt = float(orbit_dict['time']['spacing'])
    t0 = ref_epoch + isce3.core.TimeDelta(float(orbit_dict['time']['first']))
    n_pts = int(orbit_dict['time']['size'])
    orbit_sv = [[]] * n_pts
    for i in range(n_pts):
        t = t0 + isce3.core.TimeDelta(i * dt)
        pos = [float(orbit_dict[f'position_{xyz}'][i]) for xyz in 'xyz']
        vel = [float(orbit_dict[f'velocity_{xyz}'][i]) for xyz in 'xyz']
        orbit_sv[i] = isce3.core.StateVector(t, pos, vel)

    return Orbit(orbit_sv, ref_epoch)


@dataclass(frozen=True)
class GeoCslcMetadata():
    # subset of burst class attributes
    sensing_start: datetime
    sensing_stop: datetime
    radar_center_frequency: float
    wavelength: float
    azimuth_steer_rate: float
    azimuth_time_interval: float
    slant_range_time: float
    starting_range: float
    range_sampling_rate: float
    range_pixel_spacing: float
    azimuth_fm_rate: Poly1d
    doppler: Poly1d
    range_bandwidth: float
    polarization: str # {VV, VH, HH, HV}
    burst_id: str # t{track_number}_{burst_number}_iw{1,2,3}
    platform_id: str # S1{A,B}
    center: Point # {center lon, center lat} in degrees
    border: Polygon # list of lon, lat coordinate tuples (in degrees) representing burst border
    orbit: isce3.core.Orbit
    orbit_direction: str
    # VRT params
    tiff_path: str  # path to measurement tiff in SAFE/zip
    i_burst: int
    # window parameters
    range_window_type: str
    range_window_coefficient: float

    runconfig: SimpleNamespace
    geogrid: GeoGridParameters
    nodata: str
    input_data_ipf_version: str
    isce3_version: str

    @classmethod
    def from_georunconfig(cls, cfg: GeoRunConfig, burst_id: str):
        '''Create GeoBurstMetadata class from GeoRunConfig object

        Parameter:
        ---------
        cfg : GeoRunConfig
            GeoRunConfig containing geocoded burst metadata
        burst_id : str
            ID of burst to create metadata object for
        '''
        burst = None
        for b in cfg.bursts:
            if str(b.burst_id) == burst_id:
                burst = b
                date_str = burst.sensing_start.strftime("%Y%m%d")
                burst_id_date_key = (burst_id, date_str)
                out_paths = cfg.output_paths[burst_id_date_key]
                break

        if burst is None:
            err_str = f'{burst_id} not found in cfg.bursts'
            raise ValueError(err_str)

        geogrid = cfg.geogrids[burst_id]

        # get boundary from geocoded raster
        date_str = burst.sensing_start.strftime("%Y%m%d")
        pol = burst.polarization
        hdf5_path = out_paths.hdf5_path
        dataset_path_template = f'HDF5:%FILE_PATH%://complex_backscatter/{pol}'
        geo_boundary = get_boundary_polygon(hdf5_path, np.nan,
                                            dataset_path_template)
        center = geo_boundary.centroid

        # place holders
        nodata_val = 'NaN'
        ipf_ver = '?'
        isce3_ver = isce3.__version__

        return cls(burst.sensing_start, burst.sensing_stop,
                   burst.radar_center_frequency, burst.wavelength,
                   burst.azimuth_steer_rate, burst.azimuth_time_interval,
                   burst.slant_range_time, burst.starting_range,
                   burst.range_sampling_rate, burst.range_pixel_spacing,
                   burst.azimuth_fm_rate, burst.doppler.poly1d,
                   burst.range_bandwidth, burst.polarization, burst_id,
                   burst.platform_id, center, geo_boundary, burst.orbit,
                   burst.orbit_direction, burst.tiff_path, burst.i_burst,
                   burst.range_window_type, burst.range_window_coefficient,
                   cfg.groups, geogrid, nodata_val, ipf_ver, isce3_ver)


    @classmethod
    def from_file(cls, file_path: str, fmt: str):
        '''Create GeoBurstMetadata class from json file

        Parameter:
        ---------
        file_path: str
            File containing geocoded burst metadata
        '''
        if fmt == 'yaml':
            yaml = YAML(typ='safe')
            load = yaml.load
        elif fmt == 'json':
            load = json.load
        else:
            raise ValueError(f'{fmt} unsupported. Only "json" or "yaml" supported')

        with open(file_path, 'r') as fid:
            meta_dict = load(fid)

        datetime_fmt = "%Y-%m-%d %H:%M:%S.%f"
        sensing_start = datetime.strptime(meta_dict['sensing_start'],
                                          datetime_fmt)
        sensing_stop = datetime.strptime(meta_dict['sensing_stop'],
                                         datetime_fmt)

        azimuth_fm_rate = _poly1d_from_dict(meta_dict['azimuth_fm_rate'])

        dopp_poly1d = _poly1d_from_dict(meta_dict['doppler'])

        orbit = _orbit_from_dict(meta_dict['orbit'])

        # init geo_runconfig
        cfg = wrap_namespace(meta_dict['runconfig'])

        # init geogrid
        grid_dict = meta_dict['geogrid']
        geogrid = GeoGridParameters(grid_dict['start_x'], grid_dict['start_y'],
                                    grid_dict['spacing_x'],
                                    grid_dict['spacing_y'],
                                    grid_dict['length'], grid_dict['width'],
                                    grid_dict['epsg'])

        # get boundary from geocoded raster
        product_path = cfg.product_path_group.product_path
        date_str = sensing_start.strftime("%Y%m%d")
        burst_id = meta_dict['burst_id']
        pol = meta_dict['polarization']
        output_dir = f'{product_path}/{burst_id}/{date_str}'
        file_stem = f'geo_{burst_id}_{pol}'
        geo_raster_path = f'{output_dir}/{file_stem}'
        geo_boundary = get_boundary_polygon(geo_raster_path, np.nan)
        center = geo_boundary.centroid

        return cls(sensing_start, sensing_stop,
                   meta_dict['radar_center_frequency'],
                   meta_dict['wavelength'], meta_dict['azimuth_steer_rate'],
                   meta_dict['azimuth_time_interval'],
                   meta_dict['slant_range_time'], meta_dict['starting_range'],
                   meta_dict['range_sampling_rate'],
                   meta_dict['range_pixel_spacing'], azimuth_fm_rate,
                   dopp_poly1d, meta_dict['range_bandwidth'], pol,
                   meta_dict['burst_id'],  meta_dict['platform_id'],
                   center, geo_boundary, orbit, meta_dict['orbit_direction'],
                   meta_dict['tiff_path'], meta_dict['i_burst'],
                   meta_dict['range_window_type'],
                   meta_dict['range_window_coefficient'], cfg, geogrid,
                   meta_dict['nodata'], meta_dict['input_data_ipf_version'],
                   meta_dict['isce3_version'])

    def as_dict(self):
        ''' Convert self to dict for write to YAML/JSON
        '''
        self_as_dict = {}
        for key, val in self.__dict__.items():
            if key in ['border', 'center', 'sensing_start', 'sensing_stop']:
                val = str(val)
            elif isinstance(val, np.float64):
                val = float(val)
            elif key in ['azimuth_fm_rate', 'doppler']:
                temp = {}
                temp['order'] = val.order
                temp['mean'] = val.mean
                temp['std'] = val.std
                temp['coeffs'] = val.coeffs
                val = temp
            elif key == 'orbit':
                temp = {}
                temp['ref_epoch'] = str(val.reference_epoch)
                temp['time'] = {}
                temp['time']['first'] = val.time.first
                temp['time']['spacing'] = val.time.spacing
                temp['time']['last'] = val.time.last
                temp['time']['size'] = val.time.size
                temp['position_x'] = val.position[:,0].tolist()
                temp['position_y'] = val.position[:,1].tolist()
                temp['position_z'] = val.position[:,2].tolist()
                temp['velocity_x'] = val.velocity[:,0].tolist()
                temp['velocity_y'] = val.velocity[:,1].tolist()
                temp['velocity_z'] = val.velocity[:,2].tolist()
                val = temp
            elif key == 'runconfig':
                val = unwrap_to_dict(val)
            elif key == 'geogrid':
                temp = {}
                temp['start_x'] = val.start_x
                temp['start_y'] = val.start_y
                temp['spacing_x'] = val.spacing_x
                temp['spacing_y'] = val.spacing_y
                temp['length'] = val.length
                temp['width'] = val.width
                temp['epsg'] = val.epsg
                val = temp

            self_as_dict[key] = val

        return self_as_dict

    def to_file(self, dst, fmt:str):
        '''Write self to file

        Parameter:
        ---------
        dst: file pointer
            File object to write metadata to
        fmt: ['yaml', 'json']
            Format of output
        '''
        self_as_dict = self.as_dict()

        if fmt == 'yaml':
            yaml = YAML(typ='safe')
            yaml.dump(self_as_dict, dst)
        elif fmt == 'json':
            json.dump(self_as_dict, dst, indent=4)
        else:
            raise ValueError(f'{fmt} unsupported. Only "json" or "yaml" supported')

    def to_hdf5(self, dst_h5):
        '''Write self to HDF5

        Parameter:
        ---------
        dst_h5: h5py File
            HDF5 file meta data will be written to
        '''
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
                              self.isce3_version,
                              {'description': 'ISCE3 version used to generate the L2_CSLC_S1 product'})
        add_dataset_and_attrs(id_proc_group, 'project', 'OPERA', {})
        add_dataset_and_attrs(id_proc_group, 'product_level', '2', {})
        add_dataset_and_attrs(id_proc_group, 'product_type', 'CSLC_S1', {})
        add_dataset_and_attrs(id_proc_group, 'processing_datetime',
                              datetime.now().strftime('%Y-%M-%D %H:%M:%S'),
                              {'description': 'L2_CSLC_S1 product processing date and time',
                               'format': 'YYYY-MM-DDD HH:MM:SS'})
        add_dataset_and_attrs(id_proc_group, 'spacecraft_name',
                              self.platform_id,
                              {'description': 'Name of Sensor platform (e.g., S1-A/B)'})

        # subset of burst class attributes
        s1ab_group = metadata_group.require_group('s1ab_burst_metadata')
        add_dataset_and_attrs(s1ab_group, 'sensing_start',
                              self.sensing_start.strftime('%Y-%M-%D %H:%M:%S.%f'),
                              {'description': 'Sensing start time of the burst',
                               'format': 'YYYY-MM-DD HH:MM:SS.6f'})
        add_dataset_and_attrs(s1ab_group, 'sensing_stop',
                              self.sensing_stop.strftime('%Y-%M-%D %H:%M:%S.%f'),
                              {'description':'Sensing stop time of the burst',
                               'format': 'YYYY-MM-DD HH:MM:SS.6f'})
        add_dataset_and_attrs(s1ab_group, 'radar_center_frequency',
                              self.radar_center_frequency,
                              {'description':'Radar center frequency',
                               'units':'Hz'})
        add_dataset_and_attrs(s1ab_group, 'wavelength', self.wavelength,
                              {'description':'Wavelength of the transmitted signal',
                               'units':'meters'})
        add_dataset_and_attrs(s1ab_group, 'azimuth_steer_rate',
                              self.azimuth_steer_rate,
                              {'description':'Azimuth steering rate of IW and EW modes',
                               'units':'degrees/second'})
        # TODO add input width and length
        add_dataset_and_attrs(s1ab_group, 'azimuth_time_interval',
                              self.azimuth_time_interval,
                              {'description':'Time spacing between azimuth lines of the burst',
                               'units':'seconds'})
        add_dataset_and_attrs(s1ab_group, 'starting_range',
                              self.starting_range,
                              {'description':'Slant range of the first sample of the input burst',
                               'units':'meters'})
        # TODO do we need this? It's not in the specs.
        # TODO add far_range?
        add_dataset_and_attrs(s1ab_group, 'slant_range_time',
                              self.slant_range_time,
                              {'description':'two-way slant range time of Doppler centroid frequency estimate',
                               'units':'seconds'})
        add_dataset_and_attrs(s1ab_group, 'range_pixel_spacing',
                              self.range_pixel_spacing,
                              {'description':'Pixel spacing between slant range samples in the input burst SLC',
                               'units':'meters'})
        add_dataset_and_attrs(s1ab_group, 'range_bandwidth',
                              self.range_bandwidth,
                              {'description':'Slant range bandwidth of the signal',
                               'units':'Hz'})
        add_dataset_and_attrs(s1ab_group, 'polarization',
                              self.polarization,
                              {'description': 'Polarization of the burst'})
        add_dataset_and_attrs(s1ab_group, 'platform_id',
                              self.platform_id,
                              {'description': 'Sensor platform identification string (e.g., S1A or S1B)'})
        # window parameters
        add_dataset_and_attrs(s1ab_group, 'range_window_type',
                              self.range_window_type,
                              {'description': 'name of the weighting window type used during processing'})
        add_dataset_and_attrs(s1ab_group, 'range_window_coefficient',
                              self.range_window_coefficient,
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
        poly1d_to_h5(s1ab_group, 'azimuth_fm_rate', self.azimuth_fm_rate)
        poly1d_to_h5(s1ab_group, 'doppler', self.doppler)

        # save orbit
        orbit_group = metadata_group.require_group('orbit')
        ref_epoch = self.orbit.reference_epoch.isoformat().replace('T', ' ')
        add_dataset_and_attrs(orbit_group, 'ref_epoch', ref_epoch,
                              {'description': 'Reference epoch of the state vectors',
                               'format': 'YYYY-MM-DD HH:MM:SS.6f'})
        orbit_time_obj = self.orbit.time
        add_dataset_and_attrs(orbit_group, 'time',
                              np.linspace(orbit_time_obj.first,
                                          orbit_time_obj.last,
                                          orbit_time_obj.size),
                              {'description': 'Time of the orbit state vectors relative to the reference epoch',
                               'units': 'seconds'})
        for i_ax, axis in enumerate('xyz'):
            add_dataset_and_attrs(orbit_group, f'position_{axis}',
                                  self.orbit.position[:, i_ax],
                                  {'description': f'Platform position along {axis}-direction',
                                   'units': 'meters'})
            add_dataset_and_attrs(orbit_group, f'velocity_{axis}',
                                  self.orbit.velocity[:, i_ax],
                                  {'description': f'Platform velocity along {axis}-direction',
                                   'units': 'meters/second'})
        add_dataset_and_attrs(orbit_group, 'orbit_direction',
                              self.orbit_direction,
                              {'description':'Direction of sensor orbit ephermerides (e.g., ascending, descending)'})

        # VRT params
        input_group = metadata_group.require_group('input')
        add_dataset_and_attrs(input_group, 'safe_tiff_path',
                              self.tiff_path,
                              {'description': 'Path to TIFF file within the SAFE file containing the burst'})
        add_dataset_and_attrs(input_group, 'input_data_ipf_version',
                              self.input_data_ipf_version,
                              {'description': 'Instrument Processing Facility'})

        processing_group = metadata_group.require_group('processing')
        add_dataset_and_attrs(processing_group, 'burst_id',
                              self.burst_id,
                              {'description': 'Unique burst identification string (ESA convention)'})
        center_lon_lat = np.array([val[0] for val in self.center.coords.xy])
        add_dataset_and_attrs(processing_group, 'center',
                              center_lon_lat,
                              {'description': 'Burst geographical center coordinates in the projection system used for processing',
                               'units': 'meters'})
        # list of coordinate tuples (in degrees) representing burst border
        border_x, border_y = self.border.exterior.coords.xy
        border_lon_lat = np.array([[lon, lat] for lon, lat in zip(border_x,
                                                                  border_y)])
        add_dataset_and_attrs(processing_group, 'border', border_lon_lat,
                              {'description': 'X- and Y- coordinates of the polygon including valid L2_CSLC_S1 data',
                               'units': 'meters'})
        add_dataset_and_attrs(processing_group, 'i_burst', self.i_burst,
                              {'description': 'Index of the burst of interest relative to other bursts in the S1-A/B SAFE file'})
        add_dataset_and_attrs(processing_group, 'start_x', self.geogrid.start_x,
                              {'description': 'X-coordinate of the L2_CSLC_S1 starting point in the coordinate system selected for processing',
                               'units': 'meters'})
        add_dataset_and_attrs(processing_group, 'start_y', self.geogrid.start_y,
                              {'description': 'Y-coordinate of the L2_CSLC_S1 starting point in the coordinate system selected for processing',
                               'units': 'meters'})
        add_dataset_and_attrs(processing_group, 'x_posting',
                              self.geogrid.spacing_x,
                              {'description': 'Spacing between product pixels along the X-direction ',
                               'units': 'meters'})
        add_dataset_and_attrs(processing_group, 'y_posting',
                              self.geogrid.spacing_y,
                              {'description': 'Spacing between product pixels along the Y-direction ',
                               'units': 'meters'})
        add_dataset_and_attrs(processing_group, 'width', self.geogrid.width,
                              {'description': 'Number of samples in the L2_CSLC_S1 product'})
        add_dataset_and_attrs(processing_group, 'length', self.geogrid.length,
                              {'description': 'Number of lines in the L2_CSLC_S1 product'})
        add_dataset_and_attrs(processing_group, 'epsg',
                              self.geogrid.epsg,
                              {'description': 'EPSG code identifying the coordinate system used for processing'})
        add_dataset_and_attrs(processing_group, 'no_data_value',
                              self.nodata,
                              {'description': 'Value used when no data present'})

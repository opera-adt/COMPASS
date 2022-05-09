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
    burst_id: str # t{track_number}_iw{1,2,3}_b{burst_index}
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
    def from_georunconfig(cls, cfg: GeoRunConfig):
        '''Create GeoBurstMetadata class from GeoRunConfig object

        Parameter:
        ---------
        cfg : GeoRunConfig
            GeoRunConfig containing geocoded burst metadata
        '''
        burst = cfg.bursts[0]
        burst_id = burst.burst_id

        geogrid = cfg.geogrids[burst_id]

        # get boundary from geocoded raster
        burst_id = burst.burst_id
        date_str = burst.sensing_start.strftime("%Y%m%d")
        pol = burst.polarization
        geo_raster_path = f'{cfg.output_dir}/{burst_id}_{date_str}_{pol}.slc'
        geo_boundary = get_boundary_polygon(geo_raster_path, np.nan)
        center = geo_boundary.centroid

        # place holders
        nodata_val = '?'
        ipf_ver = '?'
        isce3_ver = '?'

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

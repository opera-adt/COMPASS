from dataclasses import dataclass
from datetime import datetime
import json
from types import SimpleNamespace

import isce3
from isce3.core import LUT2d, Poly1d, Orbit
from isce3.product import GeoGridParameters
import numpy as np
from s1reader.s1_burst_slc import Doppler

from compass.utils.wrap_namespace import wrap_namespace

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
class GeoBurstMetadata():
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
    doppler: Doppler
    range_bandwidth: float
    polarization: str # {VV, VH, HH, HV}
    burst_id: str # t{track_number}_iw{1,2,3}_b{burst_index}
    platform_id: str # S1{A,B}
    center: tuple # {center lon, center lat} in degrees
    border: list # list of lon, lat coordinate tuples (in degrees) representing burst border
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
    def load_from_file(cls, file_path: str):
        '''Create GeoBurstMetadata class from json file

        Parameter:
        ---------
        file_path: str
            File containing geocoded burst metadata
        '''
        with open(file_path, 'r') as fid:
            meta_dict = json.load(fid)

        fmt = "%Y-%m-%d %H:%M:%S.%f"
        sensing_start = datetime.strptime(meta_dict['sensing_start'], fmt)
        sensing_stop = datetime.strptime(meta_dict['sensing_stop'], fmt)

        azimuth_fm_rate = _poly1d_from_dict(meta_dict['azimuth_fm_rate'])

        dopp_poly1d = _poly1d_from_dict(meta_dict['doppler']['poly1d'])
        dopp_lut2d = _lut2d_from_dict(meta_dict['doppler']['lut2d'])
        doppler = Doppler(dopp_poly1d, dopp_lut2d)

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

        nodata_val = meta_dict['nodata']
        ipf_ver = meta_dict['input_data_ipf_version']
        isce3_ver = meta_dict['isce3_version']

        return cls(sensing_start, sensing_stop,
                   meta_dict['radar_center_frequency'],
                   meta_dict['wavelength'], meta_dict['azimuth_steer_rate'],
                   meta_dict['azimuth_time_interval'],
                   meta_dict['slant_range_time'], meta_dict['starting_range'],
                   meta_dict['range_sampling_rate'],
                   meta_dict['range_pixel_spacing'], azimuth_fm_rate, doppler,
                   meta_dict['range_bandwidth'], meta_dict['polarization'],
                   meta_dict['burst_id'], meta_dict['platform_id'],
                   meta_dict['center'], meta_dict['border'], orbit,
                   meta_dict['orbit_direction'], meta_dict['tiff_path'],
                   meta_dict['i_burst'], meta_dict['range_window_type'],
                   meta_dict['range_window_coefficient'],
                   cfg, geogrid, nodata_val, ipf_ver, isce3_ver)

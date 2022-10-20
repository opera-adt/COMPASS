from __future__ import annotations
from dataclasses import dataclass
import json

import isce3
from isce3.product import GeoGridParameters
import journal
from ruamel.yaml import YAML

from compass.utils.geo_grid import generate_geogrids, geogrid_as_dict
from compass.utils.runconfig import (
    runconfig_to_bursts,
    load_validate_yaml,
    RunConfig)
from compass.utils.wrap_namespace import wrap_namespace


def check_geocode_dict(geocode_cfg: dict) -> None:
    error_channel = journal.error('runconfig.check_and_prepare_geocode_params')

    # check output EPSG
    output_epsg = geocode_cfg['output_epsg']
    if output_epsg is not None:
        # check 1024 <= output_epsg <= 32767:
        if output_epsg < 1024 or 32767 < output_epsg:
            err_str = f'output epsg {output_epsg} in YAML out of bounds'
            error_channel.log(err_str)
            raise ValueError(err_str)

    for xy in 'xy':
        # check posting value in current axis
        posting_key = f'{xy}_posting'
        if geocode_cfg[posting_key] is not None:
            posting = geocode_cfg[posting_key]
            if posting <= 0:
                err_str = '{xy} posting from config of {posting} <= 0'
                error_channel.log(err_str)
                raise ValueError(err_str)

        # check snap value in current axis
        snap_key = f'{xy}_snap'
        if geocode_cfg[snap_key] is not None:
            snap = geocode_cfg[snap_key]
            if snap <= 0:
                err_str = '{xy} snap from config of {snap} <= 0'
                error_channel.log(err_str)
                raise ValueError(err_str)


@dataclass(frozen=True)
class GeoRunConfig(RunConfig):
    '''dataclass containing GCSLC runconfig'''
    # dict of geogrids associated to burst IDs
    geogrids: dict[str, GeoGridParameters]

    @classmethod
    def load_from_yaml(cls, yaml_path: str, workflow_name: str) -> GeoRunConfig:
        """Initialize RunConfig class with options from given yaml file.

        Parameters
        ----------
        yaml_path : str
            Path to yaml file containing the options to load
        workflow_name: str
            Name of the workflow for which uploading default options
        """
        cfg = load_validate_yaml(yaml_path, workflow_name)
        groups_cfg = cfg['runconfig']['groups']

        geocoding_dict = groups_cfg['processing']['geocoding']
        check_geocode_dict(geocoding_dict)

        # Convert runconfig dict to SimpleNamespace
        sns = wrap_namespace(groups_cfg)

        # Load bursts
        bursts = runconfig_to_bursts(sns)

        # Load geogrids
        dem_file = groups_cfg['dynamic_ancillary_file_group']['dem_file']
        geogrids = generate_geogrids(bursts, geocoding_dict, dem_file)

        # Empty reference dict for base runconfig class constructor
        empty_ref_dict = {}

        return cls(cfg['runconfig']['name'], sns, bursts, empty_ref_dict,
                   geogrids)

    @property
    def geocoding_params(self) -> dict:
        return self.groups.processing.geocoding

    @property
    def rdr2geo_params(self) -> dict:
        return self.groups.processing.rdr2geo

    @property
    def burst_id(self) -> str:
        return self.bursts[0].burst_id

    @property
    def sensing_start(self):
        return self.bursts[0].sensing_start

    @property
    def polarization(self) -> str:
        return self.bursts[0].polarization

    @property
    def split_spectrum_params(self) -> dict:
        return self.groups.processing.range_split_spectrum

    @property
    def output_dir(self) -> str:
        date_str = self.sensing_start.strftime("%Y%m%d")
        burst_id = self.burst_id
        return f'{super().product_path}/{burst_id}/{date_str}'

    @property
    def file_stem(self) -> str:
        burst_id = self.burst_id
        pol = self.polarization
        return f'geo_{burst_id}_{pol}'


    def as_dict(self):
        ''' Convert self to dict for write to YAML/JSON

        Unable to dataclasses.asdict() because isce3 objects can not be pickled
        '''
        # convert to dict first then dump to yaml
        self_as_dict = super().as_dict()

        self_as_dict['geogrids']= {b_id:geogrid_as_dict(geogrid)
                                   for b_id, geogrid in self.geogrids.items()}

        return self_as_dict


    def to_file(self, dst, fmt:str, add_burst_boundary=True):
        ''' Write self to file

        Parameter:
        ---------
        dst: file pointer
            File object to write metadata to
        fmt: ['yaml', 'json']
            Format of output
        add_burst_boundary: bool
            If true add burst boundary string to each burst entry in dict.
            Reads geocoded burst rasters; only viable after running s1_geocode_slc.
        '''
        self_as_dict = self.as_dict()

        self_as_dict['nodata'] = 'NO_DATA_VALUE'
        self_as_dict['input_data_ipf_version'] = '?'
        self_as_dict['isce3_version'] = isce3.__version__

        if fmt == 'yaml':
            yaml = YAML(typ='safe')
            yaml.dump(self_as_dict, dst)
        elif fmt == 'json':
            json.dumps(self_as_dict, dst, indent=4)
        else:
            raise ValueError(f'{fmt} unsupported. Only "json" or "yaml" supported')

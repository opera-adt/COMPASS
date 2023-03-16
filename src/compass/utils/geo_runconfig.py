from __future__ import annotations
from dataclasses import dataclass
import json
import yaml

import isce3
from isce3.product import GeoGridParameters
import journal
from ruamel.yaml import YAML

from compass.utils.geo_grid import (generate_geogrids_from_db,
                                    generate_geogrids, geogrid_as_dict)
from compass.utils.helpers import check_file_path
from compass.utils.runconfig import (
    create_output_paths,
    runconfig_to_bursts,
    load_validate_yaml,
    RunConfig)
from compass.utils.wrap_namespace import wrap_namespace


def check_geocode_dict(geocode_cfg: dict) -> None:
    error_channel = journal.error('runconfig.check_and_prepare_geocode_params')

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
    '''dataclass containing GSLC runconfig'''
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

        # Check TEC file if not None.
        # The ionosphere correction will be applied only if
        # the TEC file is not None.
        tec_file_path = groups_cfg['dynamic_ancillary_file_group']['tec_file']
        if tec_file_path is not None:
            check_file_path(tec_file_path)
        # Check troposphere weather model file if not None. This
        # troposphere correction is applied only if this file is not None
        weather_model_path = groups_cfg['dynamic_ancillary_file_group'][
            'weather_model_file'
        ]
        if weather_model_path is not None:
            check_file_path(weather_model_path)

        # Convert runconfig dict to SimpleNamespace
        sns = wrap_namespace(groups_cfg)

        # Load bursts
        bursts = runconfig_to_bursts(sns)

        # Load geogrids
        dem_file = groups_cfg['dynamic_ancillary_file_group']['dem_file']
        burst_database_file = groups_cfg['static_ancillary_file_group']['burst_database_file']
        if burst_database_file is None:
            geogrids = generate_geogrids(bursts, geocoding_dict, dem_file)
        else:
            geogrids = generate_geogrids_from_db(bursts, geocoding_dict,
                                                 dem_file, burst_database_file)

        # Empty reference dict for base runconfig class constructor
        empty_ref_dict = {}

        # For saving entire file with default fill-in as string to metadata.
        # Stop gap for writing dict to individual elements to HDF5 metadata
        user_plus_default_yaml_str = yaml.dump(cfg)

        # Get scratch and output paths
        output_paths = create_output_paths(sns, bursts)

        return cls(cfg['runconfig']['name'], sns, bursts, empty_ref_dict,
                   user_plus_default_yaml_str, output_paths, geogrids)

    @property
    def product_group(self) -> dict:
        return self.groups.product_path_group

    @property
    def weather_model_file(self) -> str:
        return self.groups.dynamic_ancillary_file_group.weather_model_file

    @property
    def geocoding_params(self) -> dict:
        return self.groups.processing.geocoding

    @property
    def rdr2geo_params(self) -> dict:
        return self.groups.processing.rdr2geo

    @property
    def lut_params(self) -> dict:
        return self.groups.processing.correction_luts

    @property
    def tropo_params(self) -> dict:
        return self.groups.processing.correction_luts.troposphere

    def as_dict(self):
        ''' Convert self to dict for write to YAML/JSON

        Unable to dataclasses.asdict() because isce3 objects can not be pickled
        '''
        # convert to dict first then dump to yaml
        self_as_dict = super().as_dict()

        self_as_dict['geogrids']= {b_id:geogrid_as_dict(geogrid)
                                   for b_id, geogrid in self.geogrids.items()}

        return self_as_dict


    def to_file(self, dst, fmt:str):
        ''' Write self to file

        Parameter:
        ---------
        dst: file pointer
            File object to write metadata to
        fmt: ['yaml', 'json']
            Format of output
        '''
        self_as_dict = self.as_dict()

        self_as_dict['nodata'] = 'NO_DATA_VALUE'
        self_as_dict['input_data_ipf_version'] = '?'
        self_as_dict['isce3_version'] = isce3.__version__

        if fmt == 'yaml':
            yaml_obj = YAML(typ='safe')
            yaml_obj.dump(self_as_dict, dst)
        elif fmt == 'json':
            json.dumps(self_as_dict, dst, indent=4)
        else:
            raise ValueError(f'{fmt} unsupported. Only "json" or "yaml" supported')

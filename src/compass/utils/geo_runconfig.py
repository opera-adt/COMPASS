from __future__ import annotations
from dataclasses import dataclass
import json

from isce3.product import GeoGridParameters
import journal
from ruamel.yaml import YAML

from compass.utils.geogrid import generate_geogrids, geogrid_as_dict
from compass.utils.runconfig import load_bursts, load_validate_yaml, RunConfig
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
    geogrids: dict[str:GeoGridParameters]

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
        bursts = load_bursts(sns)

        # Load geogrids
        dem_file = groups_cfg['dynamic_ancillary_file_group']['dem_file']
        geogrids = generate_geogrids(bursts, geocoding_dict, dem_file)

        return cls(cfg['runconfig']['name'], sns, bursts, geogrids)

    @property
    def geocoding_params(self) -> dict:
        return self.groups.processing.geocoding

    @property
    def split_spectrum_params(self) -> dict:
        return self.groups.processing.range_split_spectrum

    def as_dict(self):
        ''' Convert self to dict for write to YAML/JSON
        '''
        # convert to dict first then dump to yaml
        self_as_dict = super().as_dict()

        self_as_dict['geogrids']= {b_id:geogrid_as_dict(geogrid)
                                   for b_id, geogrid in self.geogrids.items()}

        return self_as_dict

    def to_yaml(self, dst):
        ''' Write self to YAML

        Parameter:
        ---------
        dst: file pointer
           Where to write YAML

        '''
        self_as_dict = self.as_dict()
        yaml = YAML(typ='safe')
        yaml.dump(self_as_dict, dst)

    def to_json(self, dst):
        ''' Write self to JSON

        Parameter:
        ---------
        dst: file pointer
           Where to write JSON

        '''
        self_as_dict = self.as_dict()
        json.dump(self_as_dict, dst)

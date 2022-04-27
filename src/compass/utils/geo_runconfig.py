from __future__ import annotations
from dataclasses import dataclass
import json

from isce3.product import GeoGridParameters
import journal
import numpy as np
from ruamel.yaml import YAML

from compass.utils.geogrid import generate_geogrids, geogrid_as_dict
from compass.utils.raster_polygon import get_boundary_polygon
from compass.utils.runconfig import (runconfig_to_bursts, load_validate_yaml,
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
        bursts = runconfig_to_bursts(sns)

        # Load geogrids
        dem_file = groups_cfg['dynamic_ancillary_file_group']['dem_file']
        geogrids = generate_geogrids(bursts, geocoding_dict, dem_file)

        # Empty reference dict for runconfig constructor
        empty_ref_dict = {}

        return cls(cfg['runconfig']['name'], sns, bursts, empty_ref_dict,
                   geogrids)

    @property
    def geocoding_params(self) -> dict:
        return self.groups.processing.geocoding

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
        '''
        # convert to dict first then dump to yaml
        self_as_dict = super().as_dict()

        self_as_dict['geogrids']= {b_id:geogrid_as_dict(geogrid)
                                   for b_id, geogrid in self.geogrids.items()}

        return self_as_dict


    def to_metadata_file(self, dst, fmt:str, add_burst_boundary=True):
        ''' Write restructured self to YAML

        Parameter:
        ---------
        dst: file pointer
            File object to write metadata to
        fmt: ['yaml', 'json']
            Format of output
        add_burst_boundary: bool
            If true add burst boundary string to each burst entry in dict.
            Reads geocoded burst rasters; only viable after running geo_cslc.
        '''
        self_as_dict = self.as_dict()

        # make burst attributes immediately accessible
        meta_dict = list(self_as_dict['bursts'].values())[0]
        meta_dict['sensing_stop'] = str(self.bursts[0].sensing_stop)

        # clear not useful burst key/vals
        meta_dict.pop('shape')
        for frst_lst in ['first', 'last']:
            for ln_smpl in ['line', 'sample']:
                key = f'{frst_lst}_valid_{ln_smpl}'
                meta_dict.pop(key)

        # add geogrid
        meta_dict['geogrid'] = self_as_dict['geogrids'][self.burst_id]

        # add runconfig groups attributes
        meta_dict['runconfig'] = self_as_dict['groups']

        if add_burst_boundary:
            # get path to geocoded raster and add to meta_dict
            geo_raster_path = f'{self.output_dir}/{self.file_stem}'
            poly = get_boundary_polygon(geo_raster_path, np.nan)
            meta_dict['poly'] = str(poly.wkt)

        meta_dict['nodata'] = 'NO_DATA_VALUE'
        meta_dict['input_data_ipf_version'] = '?'
        meta_dict['isce3_version'] = '?'

        if fmt == 'yaml':
            yaml = YAML(typ='safe')
            yaml.dump(meta_dict, dst)
        elif fmt == 'json':
            json.dump(meta_dict, dst)
        else:
            raise ValueError(f'{fmt} unsupported. Only "json" or "yaml" supported')

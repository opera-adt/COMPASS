from __future__ import annotations
from dataclasses import dataclass
import logging
import os

from ruamel.yaml import YAML
import yamale

from . import helpers


def validate_group(group_cfg: dict) -> None:
    '''check groups dict'''
    # check input group
    input_group = group_cfg['input_file_group']

    is_reference = input_group['reference_burst']['is_reference']
    if not is_reference:
        helpers.check_file_path(input_group['reference_burst']['file_path'])

    for i, safe_file in enumerate(input_group['safe_file_path']):
        # ensure only one safe file for reference rdr2geo processing
        if is_reference and i > 0:
            err_str = 'More that one safe file provided as reference'
            logging.error(err_str)
            raise ValueError(err_str)
        helpers.check_file_path(safe_file)

    helpers.check_file_path(input_group['orbit_file_path'])

    # check dynamical ancillayr file group
    dem_path = group_cfg['dynamic_ancillary_file_groups']['dem_file']
    helpers.check_file_path(dem_path)
    helpers.check_dem(dem_path)

    # check product_path_group
    product_path_group = group_cfg['product_path_group']
    helpers.check_write_dir(product_path_group['product_path'])
    helpers.check_write_dir(product_path_group['scratch_path'])
    helpers.check_file_path(product_path_group['sas_output_file'])

    # check polarizations
    if 'polarization' not in group_cfg['processing']:
        group_cfg['processing']['polarization'] = ['HH', 'HV', 'VH', 'VV']


@dataclass(frozen=True)
class RunConfig:

    '''dataclass containing CSLC runconfig'''
    name: str
    # for easy immutability, lazily keep dict read from yaml (for now?)
    groups: dict

    @classmethod
    def load_from_yaml(cls, yaml_path: str, workflow_name: str) -> RunConfig:
        '''init RunConfig from yaml file'''
        try:
            # load schema to validate against
            schema = yamale.make_schema(
                f'{helpers.WORKFLOW_SCRIPTS_DIR}/schemas/{workflow_name}.yaml',
                parser='ruamel')
        except:
            err_str = f'unable to load schema for workflow {workflow_name}.'
            logging.error(err_str)
            raise ValueError(err_str)

        # load yaml file or string from command line
        if os.path.isfile(yaml_path):
            try:
                data = yamale.make_data(yaml_path, parser='ruamel')
            except yamale.YamaleError as yamale_err:
                err_str = f'Yamale unable to load {workflow_name} runconfig yaml {yaml_path} for validation.'
                logging.error(err_str)
                raise yamale.YamaleError(err_str) from yamale_err
        else:
            raise FileNotFoundError

        # validate yaml file taken from command line
        try:
            yamale.validate(schema, data)
        except yamale.YamaleError as yamale_err:
            err_str = f'Validation fail for {workflow_name} runconfig yaml {yaml_path}.'
            logging.error(err_str)
            raise yamale.YamaleError(err_str) from yamale_err

        # load default config
        parser = YAML(typ='safe')
        default_cfg = f'{helpers.WORKFLOW_SCRIPTS_DIR}/defaults/{workflow_name}.yaml'
        with open(default_cfg, 'r') as f_default:
            cfg = parser.load(f_default)

        with open(yaml_path, 'r') as f_yaml:
            user = parser.load(f_yaml)

        # copy user suppiled config into default config
        helpers.deep_update(cfg, user)

        validate_group(cfg['groups'])

        return cls(cfg['name'], cfg['groups'])

    @property
    def burst_id(self) -> list[str]:
        return self.groups['input_file_group']['burst_id']

    @property
    def dem(self) -> str:
        return self.groups['dynamic_ancillary_file_groups']['dem_file']

    @property
    def is_reference(self) -> bool:
        return self.groups['input_file_group']['reference_burst'][
            'is_reference']

    @property
    def orbit_path(self) -> bool:
        return self.groups['input_file_group']['orbit_file_path']

    @property
    def polarization(self) -> list[str]:
        return self.groups['processing']['polarization']

    @property
    def product_path(self):
        return self.groups['product_path_group']['product_path']

    @property
    def reference_path(self) -> str:
        return self.groups['reference_burst']['file_path']

    @property
    def rdr2geo_params(self) -> dict:
        return self.groups['processing']['rdr2geo']

    @property
    def safe_files(self) -> list[str]:
        return self.groups['input_file_group']['safe_file_path']

    @property
    def sas_output_file(self):
        return self.groups['product_path_group']['sas_output_file']

    @property
    def scratch_path(self):
        return self.groups['product_path_group']['scratch_path']

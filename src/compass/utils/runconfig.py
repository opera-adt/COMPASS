from __future__ import annotations

import os
from dataclasses import dataclass

import journal
import yamale
from ruamel.yaml import YAML

from compass.utils import helpers


def validate_group(group_cfg: dict) -> None:
    """Check and validate runconfig entries.

       Parameters
       ----------
       group_cfg : dict
           Dictionary storing runconfig options to validate
    """
    error_channel = journal.error('runconfig.validate_group')

    # Check 'input_file_group' section of runconfig
    input_group = group_cfg['input_file_group']
    # If is_reference flag is False, check that file path to reference
    # burst is assigned and valid (required by geo2rdr and resample)
    is_reference = input_group['reference_burst']['is_reference']
    if not is_reference:
        helpers.check_file_path(input_group['reference_burst']['file_path'])

    for i, safe_file in enumerate(input_group['safe_file_path']):
        # ensure only one safe file for reference rdr2geo processing
        if is_reference and i > 0:
            err_str = 'More that one safe file provided as reference'
            error_channel.log(err_str)
            raise ValueError(err_str)
        helpers.check_file_path(safe_file)

    for orbit_file in input_group['orbit_file_path']:
        helpers.check_file_path(orbit_file)

    # Check 'dynamic_ancillary_file_groups' section of runconfig
    # Check that DEM file exists and is GDAL-compatible
    dem_path = group_cfg['dynamic_ancillary_file_groups']['dem_file']
    helpers.check_file_path(dem_path)
    helpers.check_dem(dem_path)

    # Check 'product_path_group' section of runconfig.
    # Check that directories herein have writing permissions
    product_path_group = group_cfg['product_path_group']
    helpers.check_write_dir(product_path_group['product_path'])
    helpers.check_write_dir(product_path_group['scratch_path'])
    helpers.check_file_path(product_path_group['sas_output_file'])

    # Check polarizations to process.
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
        """Initialize RunConfig class by loading options from
           user-defined yaml file

           Parameters
           ----------
           yaml_path : str
               Path to yaml file containing the options to load
           workflow_name: str
               Name of the workflow for which uploading default options
        """
        error_channel = journal.error('RunConfig.load_from_yaml')
        try:
            # Load schema corresponding to 'workflow_name' and to validate against
            schema = yamale.make_schema(
                f'{helpers.WORKFLOW_SCRIPTS_DIR}/schemas/cslc_s1.yaml',
                parser='ruamel')
        except:
            err_str = f'unable to load schema for workflow {workflow_name}.'
            error_channel.log(err_str)
            raise ValueError(err_str)

        # load yaml file or string from command line
        if os.path.isfile(yaml_path):
            try:
                data = yamale.make_data(yaml_path, parser='ruamel')
            except yamale.YamaleError as yamale_err:
                err_str = f'Yamale unable to load {workflow_name} runconfig yaml {yaml_path} for validation.'
                error_channel.log(err_str)
                raise yamale.YamaleError(err_str) from yamale_err
        else:
            raise FileNotFoundError

        # validate yaml file taken from command line
        try:
            yamale.validate(schema, data)
        except yamale.YamaleError as yamale_err:
            err_str = f'Validation fail for {workflow_name} runconfig yaml {yaml_path}.'
            error_channel.log(err_str)
            raise yamale.YamaleError(err_str) from yamale_err

        # load default runconfig
        parser = YAML(typ='safe')
        default_cfg_path = f'{helpers.WORKFLOW_SCRIPTS_DIR}/defaults/cslc_s1.yaml'
        with open(default_cfg_path, 'r') as f_default:
            default_cfg = parser.load(f_default)

        with open(yaml_path, 'r') as f_yaml:
            user_cfg = parser.load(f_yaml)

        # Copy user-supplied configuration options in default runconfig
        helpers.deep_update(default_cfg, user_cfg)

        validate_group(user_cfg['runconfig']['groups'])

        return cls(user_cfg['runconfig']['name'], user_cfg['runconfig']['groups'])

    @property
    def burst_id(self) -> list[str]:
        return self.groups['input_file_group']['burst_id']

    @property
    def dem(self) -> str:
        return self.groups['dynamic_ancillary_file_group']['dem_file']

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

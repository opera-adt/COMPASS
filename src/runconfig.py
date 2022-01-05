import logging
import os
from dataclasses import dataclass

import yamale
from ruamel.yaml import YAML

import helpers


def check_runconfig(runconfig_cfg: dict) -> None:
    """Check input validity of runconfig entries.
       Parameters
       ----------
       cfg: dict
        Dictionary containing runconfig parameters
    """

    # Check runconfig entries in 'input_file_group'
    input_group = runconfig_cfg['input_file_group']

    is_reference = input_group['reference_burst']['is_reference']
    # If not processing reference, check that reference burst is provided
    if not is_reference:
        helpers.check_file_path(input_group['reference_burst']['file_path'])

    # If 'is_reference' is True, ensure only one SAFE file is provided
    # as reference
    for i, safe_file in enumerate(input_group['safe_file_path']):
        if is_reference and i > 0:
            err_str = 'Multiple SAFE files provided as reference'
            logging.error(err_str)
            raise ValueError(err_str)
        # Check that provided SAFE files exists
        helpers.check_file_path(safe_file)

    # Check that provided orbit files exist
    helpers.check_file_path(input_group['orbit_file_path'])

    # Check entries in Dynamic Ancillary File group
    # Check DEM file exists and is GDAL-compatible
    dem_path = runconfig_cfg['dynamic_ancillary_file_groups']['dem_file']
    helpers.check_file_path(dem_path)
    helpers.check_dem(dem_path)

    # Check Product Path group.
    product_path_group = runconfig_cfg['product_path_group']
    helpers.check_write_dir(product_path_group['product_path'])
    helpers.check_write_dir(product_path_group['scratch_path'])
    helpers.check_file_path(product_path_group['sas_output_file'])

    # Check polarization to process. If entry is not provided,
    # process the most common used polarizations
    if 'polarization' not in runconfig_cfg['processing']:
        runconfig_cfg['processing']['polarization'] = ['HH', 'HV']


@dataclass(frozen=True)
class RunConfig:
    """Class storing runconfig entries"""
    name: str
    # for easy immutability, lazily keep dict read from yaml (for now?)
    groups: dict

    @classmethod
    def load_from_yaml(cls, yaml_path: str, workflow_name: str) -> RunConfig:
        """Initialize runconfig using user-provided YAML file
           Parameters
           ----------
           yaml_path: str
             File path to YAML file
           workflow_name: str
             Name of the workflow for which to build YAML file

           Returns:
           -------
        """
        # Upload schema for YAML file validation
        try:
            schema = yamale.make_schema(
                f'{helpers.WORKFLOW_SCRIPTS_DIR}/schemas/{workflow_name}.yaml',
                parser='ruamel')
        except ValueError:
            err_str = f'Unable to load schema for workflow {workflow_name}.'
            logging.error(err_str)
            raise ValueError(err_str)

        # Load user-provided YAML file
        if os.path.isfile(yaml_path):
            try:
                data = yamale.make_data(yaml_path, parser='ruamel')
            except yamale.YamaleError as yamale_err:
                err_str = f'Unable to load {workflow_name} yaml ' \
                          f'runconfig {yaml_path} for validation.'
                logging.error(err_str)
                raise yamale.YamaleError(err_str) from yamale_err
        else:
            raise FileNotFoundError

        # validate yaml file taken from command line
        try:
            yamale.validate(schema, data)
        except yamale.YamaleError as yamale_err:
            err_str = f'Validation fail for {workflow_name} ' \
                      f'runconfig yaml {yaml_path}.'
            logging.error(err_str)
            raise yamale.YamaleError(err_str) from yamale_err

        # Load default runconfig
        parser = YAML(typ='safe')
        default_cfg_path = f'{helpers.WORKFLOW_SCRIPTS_DIR}/' \
                           f'defaults/{workflow_name}.yaml'
        with open(default_cfg_path, 'r') as f_default:
            default_cfg = parser.load(f_default)
        # Load user-provided runconfig
        with open(yaml_path, 'r') as f_yaml:
            user_cfg = parser.load(f_yaml)

        # Copy user-supplied options in default runconfig
        helpers.deep_update(default_cfg, user_cfg)

        # Check the validity of runconfig groups and entries
        check_runconfig(default_cfg['groups'])

        return cls(default_cfg['name'], default_cfg['groups'])

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

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from itertools import cycle
import os
from types import SimpleNamespace

import journal
import yamale
from ruamel.yaml import YAML

from compass.utils import helpers
from compass.utils.wrap_namespace import wrap_namespace
from s1reader.s1_burst_slc import Sentinel1BurstSlc
from s1reader.s1_orbit import get_orbit_file_from_list
from s1reader.s1_reader import load_bursts


def validate_group_dict(group_cfg: dict) -> None:
    """Check and validate runconfig entries.

    Parameters
    ----------
    group_cfg : dict
        Dictionary storing runconfig options to validate
    """
    error_channel = journal.error('runconfig.validate_group_dict')

    # Check 'input_file_group' section of runconfig
    input_group = group_cfg['input_file_group']
    # If is_reference flag is False, check that file path to reference
    # burst is assigned and valid (required by geo2rdr and resample)
    is_reference = input_group['reference_burst']['is_reference']
    if not is_reference:
        helpers.check_file_path(input_group['reference_burst']['file_path'])

    # Check SAFE files
    run_pol_mode = group_cfg['processing']['polarization']
    safe_pol_modes = []
    for safe_file in input_group['safe_file_path']:
        # Check if files exists
        helpers.check_file_path(safe_file)

        # Get and save safe pol mode ('SV', 'SH', 'DH', 'DV')
        safe_pol_mode = helpers.get_file_polarization_mode(safe_file)
        safe_pol_modes.append(safe_pol_mode)

        # Raise error if given co-pol file and expecting cross-pol or dual-pol
        if run_pol_mode != 'co-pol' and safe_pol_mode in ['SV', 'SH']:
            err_str = f'{run_pol_mode} polarization lacks cross-pol in {safe_file}'
            error_channel.log(err_str)
            raise ValueError(err_str)

    # Check SAFE file pols consistency. i.e. no *H/*V with *V/*H respectively
    if len(safe_pol_modes) > 1:
        first_safe_pol_mode = safe_pol_modes[0][1]
        for safe_pol_mode in safe_pol_modes[1:]:
            if safe_pol_mode[1] != first_safe_pol_mode:
                err_str = 'SH/SV SAFE file mixed with DH/DV'
                error_channel.log(err_str)
                raise ValueError(err_str)

    for orbit_file in input_group['orbit_file_path']:
        helpers.check_file_path(orbit_file)

    # Check 'dynamic_ancillary_file_groups' section of runconfig
    # Check that DEM file exists and is GDAL-compatible
    dem_path = group_cfg['dynamic_ancillary_file_group']['dem_file']
    helpers.check_file_path(dem_path)
    helpers.check_dem(dem_path)

    # Check 'product_path_group' section of runconfig.
    # Check that directories herein have writing permissions
    product_path_group = group_cfg['product_path_group']
    helpers.check_write_dir(product_path_group['product_path'])
    helpers.check_write_dir(product_path_group['scratch_path'])
    helpers.check_write_dir(product_path_group['sas_output_file'])

    # Check if logging dir is writeable
    log_group = group_cfg['logging']
    helpers.check_write_dir(os.path.dirname(log_group['path']))

    write_mode = log_group['write_mode']
    journal.debug.journal.device = log_group['path']
    journal.debug.journal.device.log = open(log_group['path'], write_mode)


def get_bursts(cfg: SimpleNamespace) -> list[Sentinel1BurstSlc]:
    '''For each burst find corresponding orbit'

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration of bursts to be loaded.

    Returns
    -------
    _ : list[Sentinel1BurstSlc]
        List of bursts loaded according to given configuration.
    '''
    error_channel = journal.error('runconfig.correlate_burst_to_orbit')

    # dict to store list of bursts keyed by burst_ids
    # use default dict to handle 2 polarizations within one burst id
    bursts = defaultdict(list)

    # extract given SAFE zips to find bursts identified in cfg.burst_id
    for safe_file in cfg.input_file_group.safe_file_path:
        # from SAFE file mode, create dict of runconfig pol mode to polarization(s)
        safe_pol_mode = helpers.get_file_polarization_mode(safe_file)
        if safe_pol_mode == 'SV':
            mode_to_pols = {'co-pol':['VV']}
        elif safe_pol_mode == 'DV':
            mode_to_pols = {'co-pol':['VV'], 'cross-pol':['VH'], 'dual-pol':['VV', 'VH']}
        elif safe_pol_mode == 'SH':
            mode_to_pols = {'co-pol':['HH']}
        else:
            mode_to_pols = {'co-pol':['HH'], 'cross-pol':['HV'], 'dual-pol':['HH', 'HV']}
        pols = mode_to_pols[cfg.processing.polarization]

        # zip pol and IW subswath indices together
        i_subswaths = [1, 2, 3]
        zip_list = zip(cycle(pols), i_subswaths)

        # find orbit file
        orbit_path = get_orbit_file_from_list(
            safe_file,
            cfg.input_file_group.orbit_file_path)

        if not orbit_path:
            err_str = f"No orbit file correlates to safe file: {os.path.basename(safe_file)}"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # loop over pols and subswath index
        #import ipdb; ipdb.set_trace()
        for pol, i_subswath in zip_list:

            # loop over burst objs extracted from SAFE zip
            for burst in load_bursts(safe_file, orbit_path, i_subswath, pol):

                burst_id = burst.burst_id

                # is burst_id wanted?
                burst_id_wanted = burst_id in cfg.input_file_group.burst_id

                # does burst_id + pol exist?
                burst_id_pol_exist = False
                if burst_id in bursts:
                    if any([True for b in bursts[burst_id] if b.pol == pol]):
                        burst_id_pol_exist = True

                # add burst if wanted and doesn't already exist
                if burst_id_wanted and not burst_id_pol_exist:
                    bursts[burst_id].append(burst)

    if not bursts:
        err_str = "Could not find any of the burst IDs in the provided safe files"
        error_channel.log(err_str)
        raise ValueError(err_str)

    unaccounted_bursts = [b_id for b_id in cfg.input_file_group.burst_id
                          if b_id not in bursts]
    if unaccounted_bursts:
        err_str = f"Following burst ID(s) not found in provided safe files: {unaccounted_bursts}"
        error_channel.log(err_str)
        raise ValueError(err_str)

    return list(bursts.values())


@dataclass(frozen=True)
class RunConfig:
    '''dataclass containing CSLC runconfig'''
    # workflow name
    name: str
    # runconfig options converted from dict
    groups: SimpleNamespace
    # list of lists where bursts in interior list have a common burst_id
    bursts: list[list[Sentinel1BurstSlc]]

    @classmethod
    def load_from_yaml(cls, yaml_path: str, workflow_name: str) -> RunConfig:
        """Initialize RunConfig class with options from given yaml file.

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

        # Copy user-supplied configuration options into default runconfig
        helpers.deep_update(default_cfg, user_cfg)

        # Validate YAML values under groups dict
        validate_group_dict(default_cfg['runconfig']['groups'])

        # Convert runconfig dict to SimpleNamespace
        sns = wrap_namespace(default_cfg['runconfig']['groups'])

        bursts = get_bursts(sns)

        return cls(default_cfg['runconfig']['name'], sns, bursts)

    @property
    def burst_id(self) -> list[str]:
        return self.groups.input_file_group.burst_id

    @property
    def dem(self) -> str:
        return self.groups.dynamic_ancillary_file_group.dem_file

    @property
    def is_reference(self) -> bool:
        return self.groups.input_file_group.reference_burst.is_reference

    @property
    def orbit_path(self) -> bool:
        return self.groups.input_file_group.orbit_file_path

    @property
    def polarization(self) -> list[str]:
        return self.groups.processing.polarization

    @property
    def product_path(self):
        return self.groups.product_path_group.product_path

    @property
    def reference_path(self) -> str:
        return self.groups.reference_burst.file_path

    @property
    def rdr2geo_params(self) -> dict:
        return self.groups.processing.rdr2geo

    @property
    def safe_files(self) -> list[str]:
        return self.groups.input_file_group.safe_file_path

    @property
    def sas_output_file(self):
        return self.groups.product_path_group.sas_output_file

    @property
    def scratch_path(self):
        return self.groups.product_path_group.scratch_path

    @property
    def gpu_enabled(self):
        return self.groups.worker.gpu_enabled

    @property
    def gpu_id(self):
        return self.groups.worker.gpu_id

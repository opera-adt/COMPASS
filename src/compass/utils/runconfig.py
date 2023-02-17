from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import yaml

import isce3
import journal
import yamale
from ruamel.yaml import YAML
from s1reader.s1_burst_slc import Sentinel1BurstSlc
from s1reader.s1_orbit import get_orbit_file_from_dir
from s1reader.s1_reader import load_bursts

from compass.utils import helpers
from compass.utils.radar_grid import file_to_rdr_grid
from compass.utils.wrap_namespace import wrap_namespace, unwrap_to_dict


def load_validate_yaml(yaml_path: str, workflow_name: str) -> dict:
    """Initialize RunConfig class with options from given yaml file.

    Parameters
    ----------
    yaml_path : str
        Path to yaml file containing the options to load
    workflow_name: str
        Name of the workflow for which uploading default options

    Returns
    -------
    dict
        Validated user runconfig dict with defaults inserted
    """
    error_channel = journal.error('runconfig.load_validate_yaml')

    try:
        # Load schema corresponding to 'workflow_name' and to validate against
        schema_name = workflow_name if workflow_name == 's1_cslc_geo' \
            else 's1_cslc_radar'
        schema = yamale.make_schema(
            f'{helpers.WORKFLOW_SCRIPTS_DIR}/schemas/{schema_name}.yaml',
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
        raise FileNotFoundError(f'Yaml file {yaml_path} not found.')

    # validate yaml file taken from command line
    try:
        yamale.validate(schema, data)
    except yamale.YamaleError as yamale_err:
        err_str = f'Validation fail for {workflow_name} runconfig yaml {yaml_path}.'
        error_channel.log(err_str)
        raise yamale.YamaleError(err_str) from yamale_err

    # load default runconfig
    parser = YAML(typ='safe')
    default_cfg_path = f'{helpers.WORKFLOW_SCRIPTS_DIR}/defaults/{schema_name}.yaml'
    with open(default_cfg_path, 'r') as f_default:
        default_cfg = parser.load(f_default)

    with open(yaml_path, 'r') as f_yaml:
        user_cfg = parser.load(f_yaml)

    # Copy user-supplied configuration options into default runconfig
    helpers.deep_update(default_cfg, user_cfg)

    # Validate YAML values under groups dict
    validate_group_dict(default_cfg['runconfig']['groups'], workflow_name)

    return default_cfg


def validate_group_dict(group_cfg: dict, workflow_name) -> None:
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
    if workflow_name == 's1_cslc_radar':
        is_reference = input_group['reference_burst']['is_reference']
        if not is_reference:
            helpers.check_directory(input_group['reference_burst']['file_path'])

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


def runconfig_to_bursts(cfg: SimpleNamespace) -> list[Sentinel1BurstSlc]:
    '''Return bursts based on parameters in given runconfig

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
    bursts = []

    # extract given SAFE zips to find bursts identified in cfg.burst_id
    for safe_file in cfg.input_file_group.safe_file_path:
        # get orbit file from directory of first orbit file
        orbit_path = get_orbit_file_from_dir(
            safe_file,
            Path(cfg.input_file_group.orbit_file_path[0]).parent)

        if not orbit_path:
            err_str = f"No orbit file correlates to safe file: {os.path.basename(safe_file)}"
            error_channel.log(err_str)
            raise ValueError(err_str)

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
        pol_subswath_index_pairs = [(pol, i)
                                    for pol in pols for i in i_subswaths]

        # list of burst ID + polarization tuples
        # used to prevent reference repeats
        id_pols_found = []

        # list of burst IDs found to ensure all
        # used to ensure all IDs in config processed
        burst_ids_found = []

        # loop over pol and subswath index combinations
        for pol, i_subswath in pol_subswath_index_pairs:

            # loop over burst objs extracted from SAFE zip
            loaded_bursts =  load_bursts(safe_file, orbit_path, i_subswath, pol)
            for burst in loaded_bursts:
                # get burst ID
                burst_id = str(burst.burst_id)

                # include ALL bursts if no burst IDs given
                # is burst_id wanted? skip if not given in config
                if (cfg.input_file_group.burst_id is not None and
                        burst_id not in cfg.input_file_group.burst_id):
                    print(burst_id, cfg.input_file_group.burst_id, burst_id in cfg.input_file_group.burst_id)
                    continue

                # get polarization and save as tuple with burst ID
                pol = burst.polarization
                id_pol = (burst_id, pol)

                # has burst_id + pol combo been found?
                burst_id_pol_exist = id_pol in id_pols_found
                if not burst_id_pol_exist:
                    id_pols_found.append(id_pol)
                else:
                    continue

                # check if not a reference burst (radar grid workflow only)
                if 'reference_burst' in cfg.input_file_group.__dict__:
                    not_ref = not cfg.input_file_group.reference_burst.is_reference
                else:
                    not_ref = True

                # if not reference burst, then always ok to add
                # if reference burst, ok to add if id+pol combo does not exist
                # no duplicate id+pol combos for reference bursts
                if not_ref or not burst_id_pol_exist:
                    print(f'{burst_id} found')
                    burst_ids_found.append(burst_id)
                    bursts.append(burst)

    # check if no bursts were found
    if not bursts:
        err_str = "Could not find any of the burst IDs in the provided safe files"
        error_channel.log(err_str)
        raise ValueError(err_str)

    return bursts


def get_ref_radar_grid_info(ref_path):
    ''' Find all reference radar grids info

    Parameters
    ----------
    ref_path: str
        Path where reference radar grids processing is stored

    Returns
    -------
    ref_radar_grids:
        reference radar path and grid values found associated with
        burst ID keys
    '''
    rdr_grid_files = f'{ref_path}/radar_grid.txt'

    if not os.path.isfile(rdr_grid_files):
        raise FileNotFoundError(f'No reference radar grids not found in {ref_path}')

    ref_rdr_path = os.path.dirname(rdr_grid_files)
    ref_rdr_grid = file_to_rdr_grid(rdr_grid_files)

    return ReferenceRadarInfo(ref_rdr_path, ref_rdr_grid)


@dataclass(frozen=True)
class ReferenceRadarInfo:
    path: str
    grid: isce3.product.RadarGridParameters


def create_output_paths(sns, bursts):
    # Generate scratch and output paths
    output_paths = {}
    product_paths = sns.product_path_group
    for burst in bursts:
        # Get burst ID and check if it already
        burst_id = str(burst.burst_id)
        date_str = burst.sensing_start.strftime("%Y%m%d")

        # Key for current burst ID + date combo
        path_key = (burst_id, date_str)

        # Save output dir, output hdf5 and scratch dir to dict as
        # SimpleNamespace
        out_dir = f'{product_paths.product_path}/{burst_id}/{date_str}'
        os.makedirs(out_dir, exist_ok=True)

        fname_stem = f"{burst_id}_{date_str}"
        fname_pol = f"{fname_stem}_{burst.polarization}"
        h5_path = f"{out_dir}/{fname_stem}.h5"

        scratch_path = f'{product_paths.scratch_path}/{burst_id}/{date_str}'
        os.makedirs(scratch_path, exist_ok=True)

        output_paths[path_key] = SimpleNamespace(output_directory=out_dir,
                                                 file_name_stem=fname_stem,
                                                 file_name_pol=fname_pol,
                                                 hdf5_path=h5_path,
                                                 scratch_directory=scratch_path)
    return output_paths

@dataclass(frozen=True)
class RunConfig:
    '''dataclass containing CSLC runconfig'''
    # workflow name
    name: str
    # runconfig options converted from dict
    groups: SimpleNamespace
    # list of lists where bursts in interior list have a common burst_id
    bursts: list[Sentinel1BurstSlc]
    # dict of reference radar paths and grids values keyed on burst ID
    # (empty/unused if rdr2geo)
    reference_radar_info: ReferenceRadarInfo
    # entirety of yaml as string
    yaml_string: str
    # output paths:
    # key = tuple[burst ID, date str]
    # val = SimpleNamespace output directory path, HDF5 name, scratch directory path
    output_paths: dict[tuple[str, str], SimpleNamespace]

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
        cfg = load_validate_yaml(yaml_path, workflow_name)

        # Convert runconfig dict to SimpleNamespace
        sns = wrap_namespace(cfg['runconfig']['groups'])

        bursts = runconfig_to_bursts(sns)

        # Load reference grids if not reference run i.e. not running rdr2geo
        ref_rdr_grid_info = None
        if not sns.input_file_group.reference_burst.is_reference:
            ref_rdr_grid_info = get_ref_radar_grid_info(
                sns.input_file_group.reference_burst.file_path,
                sns.input_file_group.burst_id)

        # For saving entire file with defaults filled-in as string to metadata.
        # Stop gap for writing dict to individual elements to HDF5 metadata
        user_plus_default_yaml_str = yaml.dump(cfg)

        output_paths = create_output_paths(sns, bursts)

        return cls(cfg['runconfig']['name'], sns, bursts, ref_rdr_grid_info,
                   user_plus_default_yaml_str, output_paths)

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
        return self.groups.input_file_group.reference_burst.file_path

    @property
    def rdr2geo_params(self) -> dict:
        return self.groups.processing.rdr2geo

    @property
    def geo2rdr_params(self) -> dict:
        return self.groups.processing.geo2rdr

    @property
    def split_spectrum_params(self) -> dict:
        return self.groups.processing.range_split_spectrum

    @property
    def resample_params(self) -> dict:
        return self.groups.processing.resample

    @property
    def lut_params(self) -> dict:
        return self.groups.processing.correction_luts

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

    def as_dict(self):
        '''Convert self to dict for write to YAML/JSON

        Unable to dataclasses.asdict() because isce3 objects can not be pickled
        '''
        # Convenience functions
        def date_str(burst):
            '''Burst datetime sensing_start to str conversion
            '''
            return burst.sensing_start.date().strftime('%Y%m%d')

        def burst_as_key(burst):
            '''Create an unique key of burst ID, date string, and polarization
            '''
            return '_'.join([str(burst.burst_id), date_str(burst), burst.polarization])

        self_as_dict = {}
        for key, val in self.__dict__.items():
            if key == 'groups':
                val = unwrap_to_dict(val)
            elif key == 'bursts':
                val = {burst_as_key(burst): burst.as_dict() for burst in val}
            self_as_dict[key] = val
        return self_as_dict

    def to_yaml(self):
        '''Dump runconfig as string to sys.stdout
        '''
        self_as_dict = self.as_dict()
        yaml_obj = YAML(typ='safe')
        yaml_obj.dump(self_as_dict, sys.stdout)

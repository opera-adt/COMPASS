import re
import time

import journal

from compass import s1_geocode_metadata, s1_rdr2geo
from compass.utils.geo_runconfig import GeoRunConfig
from compass.utils.helpers import (
    bursts_grouping_generator,
    get_module_name,
    get_time_delta_str,
)
from compass.utils.yaml_argparse import YamlArgparse


def _make_rdr2geo_cfg(yaml_runconfig_str):
    """
    Make a rdr2geo specific runconfig with latitude, longitude, and height
    layers enabled for static layer product generation while preserving all
    other rdr2geo config settings

    Parameters
    ----------
    yaml_runconfig_str: str
        Workflow runconfig as a string

    Returns
    -------
    rdr2geo_cfg: dict
        Dictionary with rdr2geo longitude, latitude, and height layers
        enabled. All other rdr2geo parameters are from *yaml_runconfig_str*
    """
    # If any of the requisite layers are false, make them true in yaml cfg str
    for layer in ["latitude", "longitude", "incidence_angle"]:
        re.sub(
            f"compute_{layer}:\s+[Ff]alse", f"compute_{layer}: true", yaml_runconfig_str
        )

    # Load a GeoRunConfig from modified yaml cfg string
    rdr2geo_cfg = GeoRunConfig.load_from_yaml(
        yaml_runconfig_str, workflow_name="s1_cslc_geo"
    )

    return rdr2geo_cfg


def run(cfg: GeoRunConfig):
    """
    Run static layers workflow (i.e., generate static layers,
    geocode them, create product HDF5) with user-defined
    args stored in dictionary runconfig *cfg*

    Parameters
    ---------
    cfg: GeoRunConfig
        GeoRunConfig object with user runconfig options
    """

    module_name = get_module_name(__file__)
    info_channel = journal.info(f"{module_name}.run")
    info_channel.log(f"Starting {module_name} burst")

    # Start tracking processing time
    t_start = time.perf_counter()

    for burst_id, bursts in bursts_grouping_generator(cfg.bursts):
        burst = bursts[0]

        date_str = burst.sensing_start.strftime("%Y%m%d")

        info_channel.log(f"Starting geocoding of {burst_id} for {date_str}")

        # Generate required static layers
        rdr2geo_cfg = _make_rdr2geo_cfg(cfg.yaml_string)
        s1_rdr2geo.run(rdr2geo_cfg, burst, save_in_scratch=True)
        s1_geocode_metadata.run(cfg, burst, fetch_from_scratch=True)

    dt = get_time_delta_str(t_start)
    info_channel.log(f"{module_name} burst successfully ran in {dt} (hr:min:sec)")


def main():
    """Create the CLI and run the static layers workflow"""
    # load arguments from command line
    parser = YamlArgparse()

    # Get a runconfig dict from command line arguments
    cfg = GeoRunConfig.load_from_yaml(
        parser.run_config_path, workflow_name="s1_cslc_geo"
    )

    run(cfg)


if __name__ == "__main__":
    main()

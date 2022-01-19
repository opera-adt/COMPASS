from compass import rdr2geo
from compass.utils.runconfig import RunConfig
from compass.utils.yaml_argparse import YamlArgparse


def run(cfg):
    if cfg.is_reference:
        rdr2geo.run(cfg)


if __name__ == "__main__":
    '''run rdr2geo from command line'''
    # load command line args
    arg_parser = YamlArgparse()

    # get a runconfig dict from command line args
    runconfig = RunConfig.load_from_yaml(arg_parser.run_config_path, 'CSLC_S1')

    # run rdr2geo
    run(runconfig)

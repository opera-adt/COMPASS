from compass.utils.runconfig import RunConfig
from compass.utils.yaml_argparse import YamlArgparse


def run(cfg):
    '''
    Run geocode burst workflow with user-defined
    args stored in dictionary runconfig *cfg*

    Parameter:
    ---------
    cfg: dict
       Dictionary with user runconfig options
    '''
    print('Miao')


if __name__ == "__main__":
    '''Run geocode cslc workflow from command line'''
    # load arguments from command line
    geo_parser = YamlArgparse()

    # Get a runconfig dict from command line argumens
    runconfig = RunConfig.load_from_yaml(geo_parser.run_config_path, 'geo_cslc_s1')

    # Run geocode burst workflow
    run(runconfig)
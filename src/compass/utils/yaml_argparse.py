import argparse

class YamlArgparse():
    def __init__(self, add_grid_type=False):
        '''Initialize YamlArgparse class and parse CLI arguments for COMPASS.'''
        parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('run_config_path', type=str, nargs='?', default=None, help='Path to run config file')

        # additional arguments for s1_cslc.py
        if add_grid_type:
            parser.add_argument('-g','--grid','--grid-type', dest='grid_type', type=str,
                                default='geo', choices=['geo', 'radar'],
                                help='Grid (coordinates) type of the output CSLC')

        # parse arguments
        self.args = parser.parse_args()

    @property
    def run_config_path(self) -> str:
        return self.args.run_config_path

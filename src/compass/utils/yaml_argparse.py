import argparse

class YamlArgparse():
    def __init__(self):
        '''Initialize YamlArgparse class and parse CLI arguments for COMPASS'''
        parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('run_config_path', type=str, nargs='?', default=None, help='Path to run config file')
        self.args = parser.parse_args()

    @property
    def run_config_path(self) -> str:
        return self.args.run_config_path

import argparse

class YamlArgparse():
    def __init__(self):
        '''Initialize YamlArgparse class with basics'''
        self.parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('run_config_path', type=str, nargs='?', default=None, help='Path to run config file')

    def parse():
        '''Parse CLI arguments for COMPASS'''
        self.args = self.parser.parse_args()

    @property
    def run_config_path(self) -> str:
        return self.args.run_config_path

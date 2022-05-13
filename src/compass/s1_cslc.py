from compass import s1_rdr2geo, s1_geo2rdr, s1_resample, s1_geocode_slc
from compass.utils.geo_runconfig import GeoRunConfig
from compass.utils.runconfig import RunConfig
from compass.utils.yaml_argparse import YamlArgparse


def main():
    parser = YamlArgparse(add_grid_type=True)

    if parser.args.grid_type == 'radar':
        # CSLC workflow in radar coordinates
        # get a runconfig dict from command line args
        cfg = RunConfig.load_from_yaml(parser.run_config_path, 's1_cslc_radar')

        if cfg.is_reference:
            # reference burst - run rdr2geo and archive it
            s1_rdr2geo.run(cfg)

        else:
            # secondary burst - run geo2rdr + resample
            s1_geo2rdr.run(cfg)
            s1_resample.run(cfg)

    elif parser.args.grid_type == 'geo':
        # CSLC workflow in geo-coordinates
        # get a runconfig dict from command line argumens
        cfg = GeoRunConfig.load_from_yaml(parser.run_config_path, 's1_cslc_geo')

        # run geocode_slc
        s1_geocode_slc.run(cfg)


if __name__ == "__main__":
    '''run s1_cslc from command line'''
    main()

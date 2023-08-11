from compass import s1_static_layers
from compass.utils.geo_runconfig import GeoRunConfig


def test_geocode_slc_run(geocode_slc_params):
    """
    Run s1_geocode_slc to ensure it does not crash

    Parameters
    ----------
    geocode_slc_params: SimpleNamespace
        SimpleNamespace containing geocode SLC unit test parameters
    """
    # load yaml to cfg
    cfg = GeoRunConfig.load_from_yaml(
        geocode_slc_params.gslc_cfg_path, workflow_name="s1_cslc_geo"
    )

    # pass cfg to s1_geocode_slc
    s1_static_layers.run(cfg)

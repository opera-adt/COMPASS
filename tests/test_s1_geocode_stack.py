from pathlib import Path

from compass import s1_geocode_stack


def test_geocode_slc_run(geocode_slc_params, tmpdir):
    """
    Run s1_geocode_slc to ensure it does not crash

    Parameters
    ----------
    geocode_slc_params: SimpleNamespace
        SimpleNamespace containing geocode SLC unit test parameters
    """
    data_dir = Path(geocode_slc_params.gslc_cfg_path).parent

    with tmpdir.as_cwd():
        s1_geocode_stack.run(
            slc_dir=data_dir,
            dem_file=data_dir / "test_dem.tiff",
            orbit_dir=data_dir / "orbits",
        )

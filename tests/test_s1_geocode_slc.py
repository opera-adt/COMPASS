import csv
import pathlib

import h5py
import isce3
import numpy as np
import pytest

from compass import s1_geocode_slc
from compass.utils.geo_runconfig import GeoRunConfig


@pytest.mark.order(1)
def test_geocode_slc_run(geocode_slc_params):
    '''
    Run s1_geocode_slc to ensure it does not crash

    Parameters
    ----------
    geocode_slc_params: SimpleNamespace
        SimpleNamespace containing geocode SLC unit test parameters
    '''
    # load yaml to cfg
    cfg = GeoRunConfig.load_from_yaml(geocode_slc_params.gslc_cfg_path,
                                      workflow_name='s1_cslc_geo')

    # pass cfg to s1_geocode_slc
    s1_geocode_slc.run(cfg)


def _get_nearest_index(arr, val):
    '''
    Find index of element in given array closest to given value

    Parameters
    ----------
    arr: np.ndarray
        1D array to be searched
    val: float
        Number to be searched for

    Returns
    -------
    _: int
        Index of element in arr where val is closest
    '''
    return np.abs(arr - val).argmin()


def _get_reflectors_bounding_slice(geocode_slc_params):
    '''
    Get latitude, longitude slice that contains all the corner reflectors in
    CSV list of corner reflectors

    Parameters
    ----------
    geocode_slc_params: SimpleNamespace
        SimpleNamespace containing geocode SLC unit test parameters
    '''
    # extract from HDF5
    with h5py.File(geocode_slc_params.output_hdf5_path, 'r') as h5_obj:
        grid_group = h5_obj[geocode_slc_params.grid_group_path]

        # create projection to covert from UTM to LLH
        epsg = int(grid_group['projection'][()])
        proj = isce3.core.UTM(epsg)

        x_coords_utm = grid_group['x_coordinates'][()]
        y_coords_utm = grid_group['y_coordinates'][()]

        lons = np.array([np.degrees(proj.inverse([x, y_coords_utm[0], 0])[0])
                         for x in x_coords_utm])

        lats = np.array([np.degrees(proj.inverse([x_coords_utm[0], y, 0])[1])
                         for y in y_coords_utm])

        # get array shape for later check of slice with margins applied
        height, width = h5_obj[geocode_slc_params.raster_path].shape

    # extract all lat/lon corner reflector coordinates
    corner_lats = []
    corner_lons = []
    with open(geocode_slc_params.corner_coord_csv_path, 'r') as csvfile:
        corner_reader = csv.DictReader(csvfile)
        for row in corner_reader:
            corner_lats.append(float(row['Latitude (deg)']))
            corner_lons.append(float(row['Longitude (deg)']))

    # find nearest index for min/max of lats/lons and apply margin
    # apply margin to bounding box and ensure raster bounds are not exceeded
    # application of margin y indices reversed due descending order lats vector
    margin = 50
    i_max_y = max(_get_nearest_index(lats, np.max(corner_lats)) - margin, 0)
    i_min_y = min(_get_nearest_index(lats, np.min(corner_lats)) + margin,
                  height - 1)
    i_max_x = min(_get_nearest_index(lons, np.max(corner_lons)) + margin,
                  width - 1)
    i_min_x = max(_get_nearest_index(lons, np.min(corner_lons)) - margin, 0)

    # return as slice
    # y indices reversed to account for descending order lats vector
    return np.s_[i_max_y:i_min_y, i_min_x:i_max_x]


def test_geocode_slc_validate(geocode_slc_params):
    '''
    Check for presence of any reflectors in geocoded output

    Parameters
    ----------
    geocode_slc_params: SimpleNamespace
        SimpleNamespace containing geocode SLC unit test parameters
    '''
    # get slice where corner reflectors should be
    s_ = _get_reflectors_bounding_slice(geocode_slc_params)

    # slice raster array
    with h5py.File(geocode_slc_params.output_hdf5_path, 'r') as h5_obj:
        arr = h5_obj[geocode_slc_params.raster_path][s_]

    # check for bright spots in sliced array
    corner_reflector_threshold = 3e3
    assert np.any(np.abs(arr) > corner_reflector_threshold)


def test_metadata(geocode_slc_params):
    with h5py.File(geocode_slc_params.output_hdf5_path, 'r') as h5_obj:
        assert (h5_obj['/metadata/processing_information/inputs/dem_source'][()].decode() ==
                'DEM description was not provided.')


PARAMS_PATH = '/metadata/processing_information/parameters'


def _run_variant(geocode_slc_params, tmp_path, timing):
    '''
    Geocode the test burst again into tmp_path with extra
    processing.timing_correction keys; return the new product's path
    '''
    test_path = str(pathlib.Path(__file__).parent.resolve())
    with open(geocode_slc_params.gslc_cfg_path) as f:
        cfg_text = f.read()
    cfg_text = cfg_text.replace(f'{test_path}/product', f'{tmp_path}/product')
    cfg_text = cfg_text.replace(f'{test_path}/scratch', f'{tmp_path}/scratch')
    cfg_text = cfg_text.replace(
        '    processing:\n',
        '    processing:\n      timing_correction:\n' + ''.join(
            f'        {k}: {v}\n' for k, v in timing.items()), 1)
    cfg_path = tmp_path / 'geo_cslc_s1_timing.yaml'
    cfg_path.write_text(cfg_text)

    cfg = GeoRunConfig.load_from_yaml(str(cfg_path), workflow_name='s1_cslc_geo')
    s1_geocode_slc.run(cfg)

    rel = pathlib.Path(geocode_slc_params.output_hdf5_path).relative_to(
        f'{test_path}/product')
    return tmp_path / 'product' / rel


@pytest.mark.order(2)
def test_default_run_records_no_timing_correction(geocode_slc_params):
    with h5py.File(geocode_slc_params.output_hdf5_path, 'r') as h5:
        p = h5[PARAMS_PATH]
        assert p['azimuth_timing_correction_mode'][()].decode() == 'none'
        assert p['range_timing_correction_mode'][()].decode() == 'none'
        assert p['azimuth_timing_offset'][()] == 0.0
        assert p['range_timing_offset'][()] == 0.0
        assert p['timing_correction_source'][()].decode() == 'none'
        assert p['timing_correction_status'][()].decode() == 'not requested'
        assert p['azimuth_timing_swath_residual'][()] == 0.0
        assert p['range_timing_swath_residual'][()] == 0.0


@pytest.mark.order(2)
def test_table_mode_leaves_uncorrected_platform_identical(geocode_slc_params,
                                                          tmp_path):
    '''
    The packaged table corrects S1D only. On the S1A test burst, requesting
    the correction must change nothing in the data and say why.
    '''
    new_path = _run_variant(geocode_slc_params, tmp_path,
                            {'azimuth_mode': 'carrier_phase',
                             'range_mode': 'geometry'})
    raster = geocode_slc_params.raster_path
    carrier = f'{geocode_slc_params.grid_group_path}/azimuth_carrier_phase'
    with h5py.File(geocode_slc_params.output_hdf5_path, 'r') as ref, \
            h5py.File(new_path, 'r') as new:
        np.testing.assert_array_equal(new[raster][()], ref[raster][()])
        np.testing.assert_array_equal(new[carrier][()], ref[carrier][()])
        p = new[PARAMS_PATH]
        assert p['azimuth_timing_correction_mode'][()].decode() == \
            'carrier_phase'
        assert p['azimuth_timing_offset'][()] == 0.0
        assert p['timing_correction_source'][()].decode() == 'table'
        assert p['timing_correction_table'][()].decode() == \
            'OPERA_S1_AUX_TIMING_S1C-REF_G20261024T000000'
        assert p['timing_correction_reference_platform'][()].decode() == 'S1C'
        assert p['timing_correction_status'][()].decode() == \
            'S1A not corrected by the table'


@pytest.mark.order(2)
def test_carrier_phase_mode_changes_phase_only(geocode_slc_params, tmp_path):
    '''
    A carrier-phase offset changes the phase by exactly the carrier
    difference and nothing else: amplitude, grid and geolocation stay those of
    the uncorrected product, and the applied offset is recorded.
    '''
    dt = 46.614e-6          # S1D relative to S1C
    new_path = _run_variant(geocode_slc_params, tmp_path,
                            {'azimuth_mode': 'carrier_phase',
                             'offset_source': 'constant',
                             'azimuth_constant_offset': dt})
    ref_path = geocode_slc_params.output_hdf5_path
    grid = geocode_slc_params.grid_group_path
    with h5py.File(ref_path, 'r') as ref, h5py.File(new_path, 'r') as new:
        for coord in ('x_coordinates', 'y_coordinates'):
            np.testing.assert_array_equal(new[f'{grid}/{coord}'][()],
                                          ref[f'{grid}/{coord}'][()])
        a = ref[geocode_slc_params.raster_path][()]
        b = new[geocode_slc_params.raster_path][()]
        c_ref = ref[f'{grid}/azimuth_carrier_phase'][()]
        c_new = new[f'{grid}/azimuth_carrier_phase'][()]
        p = new[PARAMS_PATH]
        assert p['azimuth_timing_offset'][()] == pytest.approx(dt)
        assert p['timing_correction_source'][()].decode() == 'constant'
        assert p['timing_correction_status'][()].decode() == 'applied'

    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 0)
    # amplitude, hence geolocation, is the uncorrected product's
    rel_amp = np.abs(np.abs(b[ok]) - np.abs(a[ok])) / np.abs(a[ok])
    assert np.median(rel_amp) < 1e-3
    # the phase moved, by the carrier difference and nothing more
    d_carrier = np.angle(np.exp(1j * (c_new[ok] - c_ref[ok])))
    assert np.median(np.abs(d_carrier)) > 0.05
    resid = np.angle(b[ok] * np.conj(a[ok]) * np.exp(-1j * d_carrier))
    assert np.median(np.abs(resid)) < 1e-2


@pytest.mark.order(2)
def test_geometry_mode_moves_the_image(geocode_slc_params, tmp_path):
    '''
    The same offset in geometry mode resamples the data, so unlike the
    carrier-phase mode the amplitude changes: that is what tells the two modes
    apart in a product.
    '''
    dt = 46.614e-6
    new_path = _run_variant(geocode_slc_params, tmp_path,
                            {'azimuth_mode': 'geometry',
                             'offset_source': 'constant',
                             'azimuth_constant_offset': dt})
    grid = geocode_slc_params.grid_group_path
    with h5py.File(geocode_slc_params.output_hdf5_path, 'r') as ref, \
            h5py.File(new_path, 'r') as new:
        a = ref[geocode_slc_params.raster_path][()]
        b = new[geocode_slc_params.raster_path][()]
        assert new[PARAMS_PATH]['azimuth_timing_correction_mode'][()].decode() \
            == 'geometry'
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 0)
    rel_amp = np.abs(np.abs(b[ok]) - np.abs(a[ok])) / np.abs(a[ok])
    assert np.median(rel_amp) > 1e-2

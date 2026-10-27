'''
Test how GeoRunConfig loads processing.timing_correction: the table file
and bursts the table does not cover,
all of which must fail at load time rather than mid-run
'''
import dataclasses
import json

import journal
import pytest

from compass.utils import timing_correction as tc
from compass.utils.geo_runconfig import GeoRunConfig, timing_table_of

# The AUX_INS that focused the Rosamond test SAFE (S1A, 2022-10-16)
TEST_SAFE_AUX_INS = 'S1A_AUX_INS_V20190228T092500_G20211103T111906'


def write_cfg(geocode_slc_params, tmp_path, timing=None, correction_luts=None,
              static=None):
    '''Test runconfig with extra processing/static keys, written to tmp_path'''
    with open(geocode_slc_params.gslc_cfg_path) as f:
        text = f.read()
    extra = ''
    for group, keys in (('timing_correction', timing),
                        ('correction_luts', correction_luts)):
        if keys:
            extra += f'      {group}:\n' + ''.join(
                f'        {k}: {v}\n' for k, v in keys.items())
    text = text.replace('    processing:\n', '    processing:\n' + extra, 1)
    if static:
        text = text.replace(
            '    static_ancillary_file_group:\n',
            '    static_ancillary_file_group:\n' + ''.join(
                f'      {k}: {v}\n' for k, v in static.items()), 1)
    path = tmp_path / 'runconfig.yaml'
    path.write_text(text)
    return str(path)


def load(path):
    return GeoRunConfig.load_from_yaml(path, workflow_name='s1_cslc_geo')


def s1a_table(tmp_path, aux_ins):
    '''A table that corrects S1A, derived for the given AUX_INS names'''
    with open(tc.DEFAULT_TIMING_TABLE) as f:
        raw = json.load(f)
    # the current-generation entry (+46.614 us), relabelled
    raw['entries'] = raw['entries'][-1:]
    raw['entries'][0].update(platform='S1A', auxIns=aux_ins)
    path = tmp_path / 'table.json'
    path.write_text(json.dumps(raw))
    return str(path)


def test_defaults_request_nothing_and_load_no_table(geocode_slc_params,
                                                    tmp_path):
    cfg = load(write_cfg(geocode_slc_params, tmp_path))
    assert cfg.timing_params.azimuth_mode == 'none'
    assert cfg.timing_params.range_mode == 'none'
    assert cfg.timing_params.offset_source == 'table'
    assert cfg.timing_params.on_uncovered == 'error'
    assert cfg.timing_table is None


def test_table_mode_loads_packaged_table(geocode_slc_params, tmp_path):
    cfg = load(write_cfg(geocode_slc_params, tmp_path,
                         timing={'azimuth_mode': 'carrier_phase'}))
    assert cfg.timing_table.name == \
        'OPERA_S1_AUX_TIMING_S1C-REF_G20261024T000000'
    # the test burst is S1A, which the packaged table does not correct
    timing = tc.resolve_burst_timing(cfg.bursts[0], cfg.timing_params,
                                     cfg.timing_table)
    assert timing.azimuth_offset == 0.0
    assert timing.status == 'S1A not corrected by the table'


def test_table_file_from_static_ancillary(geocode_slc_params, tmp_path):
    table = s1a_table(tmp_path, [TEST_SAFE_AUX_INS])
    cfg = load(write_cfg(geocode_slc_params, tmp_path,
                         timing={'azimuth_mode': 'geometry'},
                         static={'timing_table_file': table}))
    timing = tc.resolve_burst_timing(cfg.bursts[0], cfg.timing_params,
                                     cfg.timing_table)
    assert timing.geometry_azimuth_offset == pytest.approx(46.614e-6,
                                                           abs=1e-12)


def test_uncovered_burst_fails_at_load(geocode_slc_params, tmp_path):
    table = s1a_table(tmp_path, ['S1A_AUX_INS_V20250601T000000_G20251015T084917'])
    with pytest.raises(tc.UncoveredBurstError, match=TEST_SAFE_AUX_INS):
        load(write_cfg(geocode_slc_params, tmp_path,
                       timing={'azimuth_mode': 'carrier_phase'},
                       static={'timing_table_file': table}))


def test_uncovered_burst_warn_loads(geocode_slc_params, tmp_path):
    table = s1a_table(tmp_path, ['S1A_AUX_INS_V20250601T000000_G20251015T084917'])
    cfg = load(write_cfg(geocode_slc_params, tmp_path,
                         timing={'azimuth_mode': 'carrier_phase',
                                 'on_uncovered': 'warn'},
                         static={'timing_table_file': table}))
    assert cfg.timing_table is not None


def test_missing_table_file_fails_at_load(geocode_slc_params, tmp_path):
    # COMPASS reports a missing input through journal, which raises
    with pytest.raises(journal.ApplicationError):
        load(write_cfg(geocode_slc_params, tmp_path,
                       timing={'azimuth_mode': 'geometry'},
                       static={'timing_table_file': tmp_path / 'nope.json'}))


@pytest.mark.parametrize('key,value', [('azimuth_mode', 'off'),
                                       ('range_mode', 'carrier_phase'),
                                       ('offset_source', 'aux'),
                                       ('on_uncovered', 'ignore')])
def test_schema_rejects_unknown_choices(geocode_slc_params, tmp_path, key,
                                        value):
    with pytest.raises(journal.ApplicationError) as err:
        load(write_cfg(geocode_slc_params, tmp_path, timing={key: value}))
    # journal hides the reason; the yamale error it was raised from names it
    assert f'timing_correction.{key}' in str(err.value.__context__)


def test_as_dict_names_the_table(geocode_slc_params, tmp_path):
    cfg = load(write_cfg(geocode_slc_params, tmp_path,
                         timing={'azimuth_mode': 'carrier_phase'}))
    assert cfg.as_dict()['timing_table'] == cfg.timing_table.name


def test_timing_table_of_config_built_without_load_from_yaml(
        geocode_slc_params, tmp_path):
    """cslc-batch builds GeoRunConfig directly, so timing_table is None; the
    SAS must still find the table. Regression: V5 arms failed with "no
    timing table was loaded" (2026-10-24)."""
    cfg = load(write_cfg(geocode_slc_params, tmp_path,
                         timing={'azimuth_mode': 'carrier_phase'}))
    bare = dataclasses.replace(cfg, timing_table=None)
    table = timing_table_of(bare)
    assert table.name == cfg.timing_table.name


def test_timing_table_of_keeps_the_loaded_table(geocode_slc_params, tmp_path):
    cfg = load(write_cfg(geocode_slc_params, tmp_path,
                         timing={'azimuth_mode': 'carrier_phase'}))
    assert timing_table_of(cfg) is cfg.timing_table


def test_timing_table_of_is_none_when_not_requested(geocode_slc_params,
                                                    tmp_path):
    cfg = load(write_cfg(geocode_slc_params, tmp_path))
    assert timing_table_of(dataclasses.replace(cfg, timing_table=None)) is None

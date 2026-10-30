'''
Test the relative S1 timing table: building it from ESA aux products, reading
it back, and looking bursts up in it

The AUX_INS and AUX_ITC products come from s1reader's packaged data.
'''
import datetime
import json
import pathlib
import sys
import types

import pytest

from s1reader import s1_annotation

from compass.utils import timing_correction as tc

AUX_INS_DIR = pathlib.Path(s1_annotation.__file__).parent / 'data' / 'aux_ins'
AUX_ITC_DIR = pathlib.Path(s1_annotation.__file__).parent / 'data' / 'aux_itc'

AUX_INS = {
    'S1A': 'S1A_AUX_INS_V20250601T000000_G20251015T084917',
    'S1C': 'S1C_AUX_INS_V20250224T000000_G20260604T134021',
    'S1D': 'S1D_AUX_INS_V20260319T000000_G20260603T113743',
}
AUX_ITC = {
    'S1A': 'S1A_AUX_ITC_V20160627T000000_G20230406T084701',
    'S1C': 'S1C_AUX_ITC_V20250519T094846_G20250521T125837',
    'S1D': 'S1D_AUX_ITC_V20260319T000000_G20260318T105930',
}
GENERATION = datetime.datetime(2026, 10, 23)


def aux_files(*platforms):
    '''AUX_INS + AUX_ITC paths for the given platforms'''
    files = []
    for p in platforms:
        files.append(str(AUX_INS_DIR / f'{AUX_INS[p]}.SAFE.zip'))
        files.append(str(AUX_ITC_DIR / f'{AUX_ITC[p]}.SAFE.zip'))
    return files


def s1d_table(**kwargs):
    '''The production case: S1D corrected relative to S1C'''
    return tc.build_timing_table(aux_files('S1C', 'S1D'), 'S1C', ['S1D'],
                                 GENERATION, **kwargs)


def written(tmp_path, table):
    return tc.load_timing_table(tc.write_timing_table(table, str(tmp_path)))


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #
def test_s1d_relative_to_s1c_matches_aux_chain():
    '''T_S1D - T_S1C = -66.299 - (-19.685) = -46.614 us, so the correction is
    +46.614 us. The archive ESD median for S1C-S1D is +47.48 us; this pins the
    sign the trade study confirmed on real seams.'''
    table = s1d_table()
    (entry,) = table['entries']
    assert entry['platform'] == 'S1D'
    assert entry['azimuthTimeCorrection'] == pytest.approx(46.614e-6, abs=1e-12)
    assert entry['rangeTimeCorrection'] == 0.0
    # -0.0 would read like a sign bug in a reviewed, config-controlled file
    assert '-0.0' not in json.dumps(table)


def test_s1a_relative_to_s1c_includes_itc():
    '''S1A: AUX_INS 0, AUX_ITC +6.3522 us, so T_S1A - T_S1C = +26.037 us. The
    ITC term is what brings the A-D archive residual from -8.5 to -2.2 us.'''
    table = tc.build_timing_table(aux_files('S1A', 'S1C'), 'S1C', ['S1A'],
                                  GENERATION)
    (entry,) = table['entries']
    assert entry['azimuthTimeCorrection'] == pytest.approx(-26.0372e-6,
                                                           abs=1e-12)
    # S1A rangeCalibration +0.74103 ns against S1C's 0
    assert entry['rangeTimeCorrection'] == pytest.approx(-7.4103e-10,
                                                         abs=1e-16)


def test_swst_bias_only_with_opt_in():
    '''swstBias is stored either way but summed into the range correction only
    when asked: S1D -6.4577 ns vs S1C -5.9341 ns.'''
    default = s1d_table()['entries'][0]
    with_swst = s1d_table(
        range_components=('rangeCalibration', 'swstBias'))['entries'][0]
    assert default['rangeComponents'] == ['rangeCalibration']
    assert default['components']['swstBias'] == pytest.approx(-6.4576909e-09)
    assert with_swst['rangeTimeCorrection'] == pytest.approx(
        -(-6.4576909e-09 - -5.9341056e-09), abs=1e-16)


def test_provenance_names_every_input():
    table = s1d_table()
    entry = table['entries'][0]
    assert entry['auxIns'] == [AUX_INS['S1D']]
    assert entry['auxItc'] == AUX_ITC['S1D']
    assert table['referenceAux'] == {'auxIns': AUX_INS['S1C'],
                                     'auxItc': AUX_ITC['S1C']}
    assert entry['components']['reference']['azimuthTimeBias'] == \
        pytest.approx(-19.685e-6)
    assert table['name'] == 'OPERA_S1_AUX_TIMING_S1C-REF_G20261023T000000'


def test_build_is_deterministic(tmp_path):
    '''Same inputs, same bytes: the table is reviewed and config-controlled'''
    a = tc.write_timing_table(s1d_table(), str(tmp_path))
    first = pathlib.Path(a).read_bytes()
    b = tc.write_timing_table(s1d_table(), str(tmp_path))
    assert pathlib.Path(b).read_bytes() == first


def write_aux_ins_zip(directory, name, azimuth_time_bias, swst_bias):
    '''A minimal AUX_INS .SAFE.zip: every IW channel, the given values'''
    import zipfile

    import lxml.etree as ET

    root = ET.Element('auxiliaryInstrument')
    lst = ET.SubElement(root, 'internalCalibrationParamsList')
    for swath in ('IW1', 'IW2', 'IW3'):
        for pol in ('VV', 'VH', 'HH', 'HV'):
            e = ET.SubElement(lst, 'internalCalibrationParams')
            ET.SubElement(e, 'swath').text = swath
            ET.SubElement(e, 'polarisation').text = pol
            ET.SubElement(e, 'azimuthTimeBias').text = repr(azimuth_time_bias)
            ET.SubElement(e, 'swstBias').text = repr(swst_bias)
    path = directory / f'{name}.SAFE.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(f'{name}.SAFE/data/{name[:3].lower()}-aux-ins.xml',
                   ET.tostring(root))
    return str(path)


#: S1D AUX_INS generations as SAR-MPC serves them (FINDINGS 17): two
#: commissioning files with swstBias 0, two with S1A's swstBias, then the
#: -66.299 us generations
S1D_HISTORY = [
    ('S1D_AUX_INS_V20251104T000000_G20251029T103833', 0.0, 0.0),
    ('S1D_AUX_INS_V20251104T000000_G20251217T143440', 0.0, -8.2256909e-09),
    ('S1D_AUX_INS_V20260319T000000_G20260318T105812', 0.0, -8.2256909e-09),
    ('S1D_AUX_INS_V20260319T000000_G20260415T150144', -6.6299e-05,
     -6.4576909e-09),
]


@pytest.fixture
def s1d_history(tmp_path):
    '''Every S1D generation above plus the vendored current one'''
    paths = [write_aux_ins_zip(tmp_path, *g) for g in S1D_HISTORY]
    return paths + [str(AUX_INS_DIR / f'{AUX_INS["S1D"]}.SAFE.zip')]


def history_table(s1d_history):
    files = [str(AUX_INS_DIR / f'{AUX_INS["S1C"]}.SAFE.zip'),
             str(AUX_ITC_DIR / f'{AUX_ITC["S1C"]}.SAFE.zip'),
             str(AUX_ITC_DIR / f'{AUX_ITC["S1D"]}.SAFE.zip'), *s1d_history]
    return tc.build_timing_table(files, 'S1C', ['S1D'], GENERATION)


def test_one_entry_per_distinct_generation_values(s1d_history):
    '''The IPF applies the AUX_INS bias of the generation that focused a
    product (V10), so bias-0 S1D products need a different correction from
    the -66.299 us ones: -(0 - (-19.685)) = -19.685 us against +46.614 us.'''
    entries = history_table(s1d_history)['entries']
    assert [e['auxIns'] for e in entries] == [
        ['S1D_AUX_INS_V20251104T000000_G20251029T103833'],
        ['S1D_AUX_INS_V20251104T000000_G20251217T143440',
         'S1D_AUX_INS_V20260319T000000_G20260318T105812'],
        ['S1D_AUX_INS_V20260319T000000_G20260415T150144', AUX_INS['S1D']],
    ]
    az = [e['azimuthTimeCorrection'] for e in entries]
    assert az == pytest.approx([-19.685e-6, -19.685e-6, 46.614e-6], abs=1e-12)
    assert entries[0]['components']['swstBias'] == 0.0


def test_history_table_round_trips_and_looks_up(tmp_path, s1d_history):
    '''Entries share validity but not generations, which the loader accepts,
    and each burst finds the entry for the generation that focused it'''
    table = written(tmp_path, history_table(s1d_history))
    assert len(table.entries) == 3
    old = table.lookup('S1D', WHEN, 'S1D_AUX_INS_V20251104T000000_G20251217T143440')
    new = table.lookup('S1D', WHEN, AUX_INS['S1D'])
    assert old.azimuth_time_correction == pytest.approx(-19.685e-6, abs=1e-12)
    assert new.azimuth_time_correction == pytest.approx(46.614e-6, abs=1e-12)


@pytest.mark.parametrize('kwargs,match', [
    ({'apply_platforms': ['S1C']}, 'is the reference'),
    ({'apply_platforms': []}, 'nothing to correct'),
    ({'apply_platforms': ['S1A']}, 'No AUX_INS/AUX_ITC given for S1A'),
    ({'range_components': ('rangeCalibration', 'bogus')}, 'range_components'),
    ({'range_components': ()}, 'range_components'),
])
def test_build_rejects_bad_requests(kwargs, match):
    args = dict(aux_files=aux_files('S1C', 'S1D'), reference_platform='S1C',
                apply_platforms=['S1D'], generation=GENERATION)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        tc.build_timing_table(**args)


def test_build_rejects_two_reference_generations(tmp_path):
    '''The reference total must be one value: the generation its products
    carry'''
    other = write_aux_ins_zip(
        tmp_path, 'S1C_AUX_INS_V20250224T000000_G20250324T091341', 0.0,
        -8.2256909e-09)
    with pytest.raises(ValueError, match='exactly one AUX_INS for the '
                                         'reference S1C'):
        tc.build_timing_table(aux_files('S1C', 'S1D') + [other], 'S1C',
                              ['S1D'], GENERATION)


@pytest.mark.parametrize('extra,match', [
    (str(AUX_INS_DIR / 'S1D_AUX_INS_V20260319T000000_G20260603T113743.SAFE.zip'),
     'given twice'),
    (str(AUX_ITC_DIR / 'S1D_AUX_ITC_V20260319T000000_G20260318T105930.SAFE.zip'),
     'Two AUX_ITC products for S1D'),
])
def test_build_rejects_duplicate_inputs(extra, match):
    with pytest.raises(ValueError, match=match):
        tc.build_timing_table(aux_files('S1C', 'S1D') + [extra], 'S1C',
                              ['S1D'], GENERATION)


@pytest.mark.parametrize('name', ['S1D_AUX_CAL_V20260319T000000_G20260318T105930',
                                  'S1D_AUX_INS_V2026_G2026', 'random.zip'])
def test_parse_aux_name_rejects_other_products(name):
    with pytest.raises(ValueError, match='Not an S1 AUX_INS/AUX_ITC'):
        tc.parse_aux_name(name)


def test_parse_aux_name_strips_suffixes():
    parts = tc.parse_aux_name(f'/x/{AUX_ITC["S1D"]}.SAFE.zip')
    assert parts['platform'] == 'S1D'
    assert parts['type'] == 'ITC'
    assert parts['name'] == AUX_ITC['S1D']
    assert parts['generation'] == datetime.datetime(2026, 3, 18, 10, 59, 30)


# --------------------------------------------------------------------------- #
# reading back
# --------------------------------------------------------------------------- #
def test_round_trip(tmp_path):
    table = written(tmp_path, s1d_table())
    assert table.name == 'OPERA_S1_AUX_TIMING_S1C-REF_G20261023T000000'
    assert table.generation == GENERATION
    assert table.reference_platform == 'S1C'
    assert table.platforms == ['S1D']
    (entry,) = table.entries
    assert entry.azimuth_time_correction == pytest.approx(46.614e-6, abs=1e-12)
    assert entry.aux_ins == (AUX_INS['S1D'],)
    assert entry.validity_start is None and entry.validity_stop is None


def raw_table(tmp_path, mutate):
    '''Write a valid table, let ``mutate`` break it, return its path'''
    raw = s1d_table()
    mutate(raw)
    path = tmp_path / 'table.json'
    path.write_text(json.dumps(raw))
    return str(path)


@pytest.mark.parametrize('mutate,match', [
    (lambda t: t.update(productType='OTHER'), 'productType'),
    (lambda t: t.pop('referencePlatform'), 'lacks referencePlatform'),
    (lambda t: t['entries'][0].pop('auxIns'), r'entries\[0\] lacks auxIns'),
    (lambda t: t['entries'][0].update(auxIns=[]), 'auxIns must be'),
    (lambda t: t['entries'][0].update(azimuthTimeCorrection='46us'),
     'must be a number'),
    (lambda t: t['entries'][0].update(azimuthTimeCorrection=True),
     'must be a number'),
    (lambda t: t['entries'][0].update(platform='Sentinel-1D'),
     'platform must look like'),
    (lambda t: t['entries'][0].update(validityStart='2026-07-01'),
     'validityStart must be'),
    (lambda t: t['entries'][0].update(validityStart='2026-07-01T00:00:00Z',
                                      validityStop='2026-06-01T00:00:00Z'),
     'not after validityStart'),
    (lambda t: t['entries'][0].update(rangeComponents=['bogus']),
     'rangeComponents'),
    (lambda t: t['entries'][0].update(platform='S1C'),
     'S1C is the reference platform'),
    (lambda t: t['entries'].append(dict(t['entries'][0])),
     'claim .* with overlapping validity'),
])
def test_load_rejects_invalid_tables(tmp_path, mutate, match):
    with pytest.raises(ValueError, match=match):
        tc.load_timing_table(raw_table(tmp_path, mutate))


def test_load_rejects_non_json(tmp_path):
    path = tmp_path / 'table.json'
    path.write_text('not json')
    with pytest.raises(ValueError, match='is not JSON'):
        tc.load_timing_table(str(path))


def test_load_accepts_adjacent_windows(tmp_path):
    '''Two S1D entries that meet but do not overlap, e.g. either side of an
    instrument event, are a valid table.'''
    def split(t):
        late = dict(t['entries'][0])
        t['entries'][0]['validityStop'] = '2026-07-15T00:00:00Z'
        late['validityStart'] = '2026-07-15T00:00:00Z'
        t['entries'].append(late)
    table = tc.load_timing_table(raw_table(tmp_path, split))
    assert len(table.entries) == 2


# --------------------------------------------------------------------------- #
# lookup
# --------------------------------------------------------------------------- #
WHEN = datetime.datetime(2026, 8, 1, 13, 55)


@pytest.mark.parametrize('platform', ['S1C', 'S1A', 'S1B'])
def test_lookup_uncorrected_platform_is_none(tmp_path, platform):
    '''The reference, and platforms deliberately left out, get no entry: zero
    correction by policy, not an error.'''
    table = written(tmp_path, s1d_table())
    assert table.lookup(platform, WHEN, 'whatever') is None


def test_lookup_finds_covered_burst(tmp_path):
    table = written(tmp_path, s1d_table())
    entry = table.lookup('S1D', WHEN, AUX_INS['S1D'])
    assert entry.azimuth_time_correction == pytest.approx(46.614e-6, abs=1e-12)


@pytest.mark.parametrize('aux_ins_name', [
    'S1D_AUX_INS_V20260319T000000_G20270101T000000', None])
def test_lookup_refuses_other_aux_ins_generation(tmp_path, aux_ins_name):
    '''A generation the entry was not derived for -- ESA revised the file, or
    the IPF started applying the bias -- must not be corrected silently.'''
    table = written(tmp_path, s1d_table())
    with pytest.raises(tc.UncoveredBurstError, match='entries were derived '
                                                     'for'):
        table.lookup('S1D', WHEN, aux_ins_name)


def test_lookup_respects_validity(tmp_path):
    start = datetime.datetime(2026, 7, 15)
    table = written(tmp_path, s1d_table(validity={'S1D': (start, None)}))
    assert table.lookup('S1D', start, AUX_INS['S1D']) is not None
    with pytest.raises(tc.UncoveredBurstError, match='no S1D entry is valid'):
        table.lookup('S1D', start - datetime.timedelta(seconds=1),
                     AUX_INS['S1D'])


def test_validity_stop_is_exclusive():
    stop = datetime.datetime(2026, 7, 15)
    entry = tc.TimingEntry('S1D', None, stop, 0.0, 0.0, ('x',), None)
    assert entry.covers_time(stop - datetime.timedelta(microseconds=1))
    assert not entry.covers_time(stop)


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #
def test_cli_builds_the_production_table(tmp_path, capsys):
    tc.main([*aux_files('S1C', 'S1D'), '--reference', 'S1C',
             '--apply', 'S1D', '--generation', '2026-10-23',
             '--validity', 'S1D=2026-07-15', '-o', str(tmp_path)])
    path = capsys.readouterr().out.strip()
    table = tc.load_timing_table(path)
    assert table.entries[0].validity_start == datetime.datetime(2026, 7, 15)
    assert table.entries[0].azimuth_time_correction == pytest.approx(
        46.614e-6, abs=1e-12)


def test_cli_takes_every_generation(tmp_path, s1d_history, capsys):
    files = [str(AUX_INS_DIR / f'{AUX_INS["S1C"]}.SAFE.zip'),
             str(AUX_ITC_DIR / f'{AUX_ITC["S1C"]}.SAFE.zip'),
             str(AUX_ITC_DIR / f'{AUX_ITC["S1D"]}.SAFE.zip'), *s1d_history]
    out = tmp_path / 'out'
    out.mkdir()
    tc.main([*files, '--reference', 'S1C', '--apply', 'S1D',
             '--generation', '2026-10-23', '-o', str(out)])
    path = capsys.readouterr().out.strip()
    assert len(tc.load_timing_table(path).entries) == 3


def test_cli_success_exits_zero(tmp_path):
    '''The console script runs sys.exit(main()); a return value there fails'''
    argv = [*aux_files('S1C', 'S1D'), '--reference', 'S1C', '--apply', 'S1D',
            '--generation', '2026-10-23', '-o', str(tmp_path)]
    with pytest.raises(SystemExit) as err:
        sys.exit(tc.main(argv))
    assert err.value.code is None


@pytest.mark.parametrize('argv', [
    ['--reference', 'S1C', '--apply', 'S1D'],           # no aux files
    ['x.zip', '--apply', 'S1D'],                        # no --reference
    ['x.zip', '--reference', 'S1C', '--apply', 'S1D',
     '--validity', 'S1D'],                              # malformed window
    ['x.zip', '--reference', 'S1C', '--apply', 'S1D',
     '--range-components', 'bogus'],
])
def test_cli_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit) as err:
        tc.main(argv)
    assert err.value.code == 2


def test_cli_help():
    with pytest.raises(SystemExit) as err:
        tc.main(['--help'])
    assert err.value.code == 0


# --------------------------------------------------------------------------- #
# per-burst resolution
# --------------------------------------------------------------------------- #
def burst(platform='S1D', aux_ins_name=AUX_INS['S1D'], sensing_start=WHEN,
          swath_name='iw2'):
    '''The four attributes resolve_burst_timing reads off a burst'''
    return types.SimpleNamespace(platform_id=platform,
                                 aux_ins_name=aux_ins_name,
                                 sensing_start=sensing_start,
                                 swath_name=swath_name)


def params(**kwargs):
    '''processing.timing_correction with the shipped defaults'''
    p = dict(azimuth_mode='none', range_mode='none', offset_source='table',
             azimuth_constant_offset=0.0, range_constant_offset=0.0,
             on_uncovered='error')
    p.update(kwargs)
    return types.SimpleNamespace(**p)


@pytest.fixture
def table(tmp_path):
    return written(tmp_path, s1d_table())


def test_resolve_defaults_request_nothing(table):
    timing = tc.resolve_burst_timing(burst(), params(), table)
    assert timing == tc.BurstTiming()
    assert timing.status == 'not requested'


def test_resolve_carrier_phase_from_table(table):
    timing = tc.resolve_burst_timing(
        burst(), params(azimuth_mode='carrier_phase'), table)
    assert timing.azimuth_offset == pytest.approx(46.614e-6, abs=1e-12)
    assert timing.carrier_phase_azimuth_offset == timing.azimuth_offset
    assert timing.geometry_azimuth_offset == 0.0
    assert timing.geometry_range_offset_meters == 0.0
    assert timing.source == 'table'
    assert timing.table_name == table.name
    assert timing.reference_platform == 'S1C'
    assert timing.status == 'applied'


def test_resolve_geometry_routes_to_the_lut(table):
    timing = tc.resolve_burst_timing(
        burst(), params(azimuth_mode='geometry'), table)
    assert timing.geometry_azimuth_offset == pytest.approx(46.614e-6,
                                                           abs=1e-12)
    assert timing.carrier_phase_azimuth_offset == 0.0


def test_resolve_axes_are_independent(table):
    '''Asking for range only must not apply the azimuth correction'''
    timing = tc.resolve_burst_timing(
        burst(), params(range_mode='geometry'), table)
    assert timing.azimuth_mode == 'none'
    assert timing.azimuth_offset == 0.0
    assert timing.range_mode == 'geometry'
    assert timing.range_offset == 0.0    # S1D vs S1C ITC range is 0 today


def test_range_offset_in_meters_is_half_the_two_way_light_time():
    timing = tc.BurstTiming(range_mode='geometry', range_offset=2e-9)
    assert timing.geometry_range_offset_meters == pytest.approx(
        299792458.0 * 1e-9)
    assert tc.BurstTiming(range_offset=2e-9).geometry_range_offset_meters == 0


def test_resolve_platform_not_in_table(table):
    timing = tc.resolve_burst_timing(
        burst('S1C', AUX_INS['S1C']), params(azimuth_mode='carrier_phase'),
        table)
    assert timing.azimuth_offset == 0.0
    assert timing.azimuth_mode == 'carrier_phase'
    assert timing.status == 'S1C not corrected by the table'


def test_resolve_uncovered_errors_by_default(table):
    with pytest.raises(tc.UncoveredBurstError, match='derived for'):
        tc.resolve_burst_timing(burst(aux_ins_name='S1D_AUX_INS_other'),
                                params(azimuth_mode='carrier_phase'), table)


def test_resolve_uncovered_warn_applies_nothing(table):
    timing = tc.resolve_burst_timing(
        burst(aux_ins_name='S1D_AUX_INS_other'),
        params(azimuth_mode='carrier_phase', on_uncovered='warn'), table)
    assert timing.azimuth_offset == 0.0
    assert timing.status.startswith('uncovered, not applied')


def test_resolve_constant_ignores_table():
    timing = tc.resolve_burst_timing(
        burst('S1A'), params(azimuth_mode='geometry', range_mode='geometry',
                             offset_source='constant',
                             azimuth_constant_offset=1e-5,
                             range_constant_offset=3e-9))
    assert timing.azimuth_offset == 1e-5
    assert timing.range_offset == 3e-9
    assert timing.source == 'constant'
    assert timing.table_name is None


def test_resolve_constant_only_for_requested_axis():
    timing = tc.resolve_burst_timing(
        burst(), params(azimuth_mode='carrier_phase', offset_source='constant',
                        azimuth_constant_offset=1e-5,
                        range_constant_offset=3e-9))
    assert timing.range_offset == 0.0


def test_resolve_table_source_without_table():
    with pytest.raises(ValueError, match='no timing table was loaded'):
        tc.resolve_burst_timing(burst(), params(azimuth_mode='geometry'))


def test_packaged_table_is_the_s1d_production_table():
    '''S1D relative to S1C, built from the -66.299 us AUX_INS generations only:
    every distributed S1D SLC (first acquisitions 2026-04-16) names one of
    them. The bias-0 commissioning generations are left out on purpose, so a
    product focused with one stops as uncovered rather than getting an
    untested correction.'''
    table = tc.load_timing_table(tc.DEFAULT_TIMING_TABLE)
    assert table.reference_platform == 'S1C'
    assert table.platforms == ['S1D']
    (entry,) = table.entries
    assert entry.azimuth_time_correction == pytest.approx(46.614e-6, abs=1e-12)
    assert entry.range_time_correction == 0.0
    assert len(entry.aux_ins) == 4 and AUX_INS['S1D'] in entry.aux_ins
    with pytest.raises(tc.UncoveredBurstError):
        table.lookup('S1D', WHEN,
                     'S1D_AUX_INS_V20260319T000000_G20260318T105812')


# --------------------------------------------------------------------------- #
# per-swath residuals
# --------------------------------------------------------------------------- #
RESIDUALS = {'S1D': {'IW1': {'azimuth': 0.4e-6},
                     'IW3': {'azimuth': -0.2e-6, 'range': 1e-10}}}
RESIDUAL_SOURCE = 'ESD median, frame 46290, test'


def residual_table(tmp_path):
    return written(tmp_path, s1d_table(swath_residuals=RESIDUALS,
                                       swath_residual_source=RESIDUAL_SOURCE))


def test_residuals_ride_on_the_entry(tmp_path):
    (entry,) = residual_table(tmp_path).entries
    assert entry.azimuth_time_correction == pytest.approx(46.614e-6,
                                                          abs=1e-12)
    assert entry.swath_residual_source == RESIDUAL_SOURCE
    assert entry.residual('iw1') == (0.4e-6, 0.0)
    assert entry.residual('IW3') == (-0.2e-6, 1e-10)
    assert entry.residual('iw2') == (0.0, 0.0)


def test_residuals_attach_to_every_generation_entry(s1d_history):
    files = [str(AUX_INS_DIR / f'{AUX_INS["S1C"]}.SAFE.zip'),
             str(AUX_ITC_DIR / f'{AUX_ITC["S1C"]}.SAFE.zip'),
             str(AUX_ITC_DIR / f'{AUX_ITC["S1D"]}.SAFE.zip'), *s1d_history]
    table = tc.build_timing_table(files, 'S1C', ['S1D'], GENERATION,
                                  swath_residuals=RESIDUALS,
                                  swath_residual_source=RESIDUAL_SOURCE)
    assert all(e['swathResiduals'] == RESIDUALS['S1D']
               for e in table['entries'])


def test_no_residuals_writes_no_residual_keys():
    """Tables without residuals stay exactly as before"""
    (entry,) = s1d_table()['entries']
    assert 'swathResiduals' not in entry
    assert 'swathResidualSource' not in entry


@pytest.mark.parametrize('kwargs,match', [
    ({'swath_residuals': RESIDUALS}, 'need a swath_residual_source'),
    ({'swath_residuals': {'S1A': RESIDUALS['S1D']},
      'swath_residual_source': 'x'}, 'which are not corrected'),
    ({'swath_residuals': {'S1D': {'IW4': {'azimuth': 1e-6}}},
      'swath_residual_source': 'x'}, 'swath must be one of'),
    ({'swath_residuals': {'S1D': {'IW1': {'elevation': 1e-6}}},
      'swath_residual_source': 'x'}, 'axis must be one of'),
    ({'swath_residuals': {'S1D': {'IW1': {'azimuth': '0.4us'}}},
      'swath_residual_source': 'x'}, 'must be a number'),
])
def test_build_rejects_bad_residuals(kwargs, match):
    with pytest.raises(ValueError, match=match):
        s1d_table(**kwargs)


@pytest.mark.parametrize('mutate,match', [
    (lambda t: t['entries'][0].update(swathResiduals=RESIDUALS['S1D']),
     'need a swathResidualSource'),
    (lambda t: t['entries'][0].update(swathResiduals={'IW9': {'azimuth': 1}},
                                      swathResidualSource='x'),
     'swath must be one of'),
])
def test_load_rejects_bad_residuals(tmp_path, mutate, match):
    with pytest.raises(ValueError, match=match):
        tc.load_timing_table(raw_table(tmp_path, mutate))


def test_resolve_adds_the_swath_residual(tmp_path):
    table = residual_table(tmp_path)
    timing = tc.resolve_burst_timing(
        burst(swath_name='iw1'),
        params(azimuth_mode='carrier_phase', range_mode='geometry'), table)
    assert timing.azimuth_offset == pytest.approx(46.614e-6 + 0.4e-6,
                                                  abs=1e-12)
    assert timing.azimuth_swath_residual == pytest.approx(0.4e-6)
    assert timing.range_offset == 0.0


def test_resolve_swath_without_residual_gets_the_aux_value(tmp_path):
    timing = tc.resolve_burst_timing(
        burst(swath_name='iw2'), params(azimuth_mode='carrier_phase'),
        residual_table(tmp_path))
    assert timing.azimuth_offset == pytest.approx(46.614e-6, abs=1e-12)
    assert timing.azimuth_swath_residual == 0.0


def test_resolve_residual_only_on_requested_axes(tmp_path):
    """IW3 has a range residual, but range_mode none applies no range"""
    timing = tc.resolve_burst_timing(
        burst(swath_name='iw3'), params(azimuth_mode='carrier_phase'),
        residual_table(tmp_path))
    assert timing.azimuth_offset == pytest.approx(46.614e-6 - 0.2e-6,
                                                  abs=1e-12)
    assert timing.range_offset == 0.0
    assert timing.range_swath_residual == 0.0


def test_cli_reads_swath_residuals(tmp_path, capsys):
    spec = tmp_path / 'residuals.json'
    spec.write_text(json.dumps({'source': RESIDUAL_SOURCE, **RESIDUALS}))
    out = tmp_path / 'out'
    out.mkdir()
    tc.main([*aux_files('S1C', 'S1D'), '--reference', 'S1C',
             '--apply', 'S1D', '--generation', '2026-10-24',
             '--swath-residuals', str(spec), '-o', str(out)])
    path = capsys.readouterr().out.strip()
    (entry,) = tc.load_timing_table(path).entries
    assert entry.swath_residual_source == RESIDUAL_SOURCE
    assert entry.residual('iw1') == (0.4e-6, 0.0)

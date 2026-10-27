'''
Relative Sentinel-1 instrument timing table (OPERA_S1_AUX_TIMING)

ESA publishes each platform's timing errors in two auxiliary products: the
AUX_INS azimuthTimeBias / swstBias and the AUX_ITC azimuthCalibration /
rangeCalibration. The IPF does not apply them to the azimuth carrier, so
burst-overlap phase between platforms carries their difference.

This module turns those products into a small JSON table of corrections
*relative to one reference platform*, so that new products from one platform
(e.g. S1D) can be aligned with an archive that is left as it is (e.g. S1C).
With T_X the summed timing error of platform X, the correction is

    c_X = -(T_X - T_ref)

in seconds, added to the burst's azimuth (or two-way range) time. Platforms
not listed in the table are not corrected.

The IPF applies the AUX_INS azimuthTimeBias of the generation that focused a
product (V10, 00_tools/tests/cslc FINDINGS 18), so T_X is per AUX_INS
generation, not per platform: a platform gets one entry per distinct set of
aux values, each naming the generations it covers. A burst focused with a
generation no entry names is "uncovered" and is not corrected silently.

The SAS only reads tables (``load_timing_table``, ``TimingTable.lookup``) and
turns the runconfig ``processing.timing_correction`` group into one decision
per burst (``resolve_burst_timing``); ``build_timing_table`` and ``main`` are
offline tools that make tables.
'''
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
from dataclasses import dataclass, field

import journal

PRODUCT_TYPE = 'OPERA_S1_AUX_TIMING'
CONVENTION = ('seconds; added to the burst azimuth time (range: two-way range '
              'time); correction = -(T_platform - T_reference)')

# Components that may be summed into the range correction. swstBias is left
# out by default: it has been nonzero since S1A, so the IPF almost certainly
# applies it at focusing and adding it again would double-count.
RANGE_COMPONENTS = ('rangeCalibration', 'swstBias')
DEFAULT_RANGE_COMPONENTS = ('rangeCalibration',)

# The table shipped with COMPASS, used when the runconfig names none
DEFAULT_TIMING_TABLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
    'OPERA_S1_AUX_TIMING_S1C-REF_G20261024T000000.json')

# Runconfig choices. 'none' rather than 'off': YAML 1.1 reads off as False.
AZIMUTH_MODES = ('none', 'carrier_phase', 'geometry')
RANGE_MODES = ('none', 'geometry')

_SPEED_OF_LIGHT = 299792458.0

# Swaths a residual may name, and what it may hold
SWATHS = ('IW1', 'IW2', 'IW3')
_RESIDUAL_AXES = ('azimuth', 'range')

# IW channels whose values must agree before one instrument-wide number is used
_IW_CHANNELS = [(swath, pol) for swath in ('iw1', 'iw2', 'iw3')
                for pol in ('vv', 'vh', 'hh', 'hv')]

_AUX_NAME = re.compile(
    r'^(?P<platform>S1[A-Z])_AUX_(?P<type>INS|ITC)_'
    r'V(?P<validity>\d{8}T\d{6})_G(?P<generation>\d{8}T\d{6})$')


class UncoveredBurstError(ValueError):
    '''A burst whose platform the table corrects, but not for this burst'''


@dataclass(frozen=True)
class TimingEntry:
    '''
    Correction for one platform over one validity window

    Attributes
    ----------
    platform: str
        Mission identifier, e.g. 'S1D'
    validity_start, validity_stop: datetime.datetime or None
        Sensing-time window the entry applies to (UTC); None leaves that end
        open
    azimuth_time_correction: float
        Correction added to the azimuth time [s]
    range_time_correction: float
        Correction added to the two-way range time [s]
    aux_ins: tuple of str
        AUX_INS product names (no .SAFE) the entry was derived for
    aux_itc: str or None
        AUX_ITC product name the entry was derived from
    components: dict
        The raw aux values used, for provenance
    range_components: tuple of str
        Which of RANGE_COMPONENTS were summed into the range correction
    swath_residuals: dict
        {'IW1': {'azimuth': s, 'range': s}, ...}: measured per-swath residuals
        [s] added on top of the aux-derived correction (e.g. ESD medians); a
        swath or axis left out adds 0
    swath_residual_source: str or None
        Where the residuals came from; required when there are any
    '''
    platform: str
    validity_start: datetime.datetime | None
    validity_stop: datetime.datetime | None
    azimuth_time_correction: float
    range_time_correction: float
    aux_ins: tuple
    aux_itc: str | None
    components: dict = field(default_factory=dict, compare=False)
    range_components: tuple = DEFAULT_RANGE_COMPONENTS
    swath_residuals: dict = field(default_factory=dict, compare=False)
    swath_residual_source: str | None = None

    def residual(self, swath):
        '''(azimuth, range) residual [s] for a swath such as 'iw2'; 0 if none'''
        values = self.swath_residuals.get(str(swath).upper(), {})
        return (float(values.get('azimuth', 0.0)),
                float(values.get('range', 0.0)))

    def covers_time(self, sensing_time):
        '''True if sensing_time falls in [validity_start, validity_stop)'''
        if self.validity_start is not None and sensing_time < self.validity_start:
            return False
        if self.validity_stop is not None and sensing_time >= self.validity_stop:
            return False
        return True


@dataclass(frozen=True)
class TimingTable:
    '''
    A relative timing table

    Attributes
    ----------
    name: str
        Product name (file name without .json), recorded in output metadata
    generation: datetime.datetime
        When the table was made (UTC)
    reference_platform: str
        Platform the corrections are relative to; its correction is zero
    entries: tuple of TimingEntry
    '''
    name: str
    generation: datetime.datetime
    reference_platform: str
    entries: tuple

    @property
    def platforms(self):
        '''Platforms the table corrects'''
        return sorted({e.platform for e in self.entries})

    def lookup(self, platform, sensing_time, aux_ins_name):
        '''
        Find the entry for one burst

        Parameters
        ----------
        platform: str
            Burst mission identifier, e.g. 'S1D'
        sensing_time: datetime.datetime
            Burst sensing start (UTC)
        aux_ins_name: str or None
            AUX_INS the IPF used, as recorded in manifest.safe

        Returns
        -------
        TimingEntry or None
            None when the table does not correct this platform at all (the
            reference platform, or one deliberately left out): zero correction
            by policy.

        Raises
        ------
        UncoveredBurstError
            The table corrects this platform but no entry covers the burst's
            sensing time, or the burst was focused with an AUX_INS generation
            no entry was derived for.
        '''
        candidates = [e for e in self.entries if e.platform == platform]
        if not candidates:
            return None

        in_time = [e for e in candidates if e.covers_time(sensing_time)]
        if not in_time:
            raise UncoveredBurstError(
                f'{self.name}: no {platform} entry is valid at {sensing_time}')

        for entry in in_time:
            if aux_ins_name in entry.aux_ins:
                return entry
        known = sorted(n for e in in_time for n in e.aux_ins)
        raise UncoveredBurstError(
            f'{self.name}: {platform} burst focused with AUX_INS '
            f'{aux_ins_name}, but the entries were derived for '
            f'{", ".join(known)}')


def parse_aux_name(path):
    '''
    Split an ESA aux product name into its parts

    Parameters
    ----------
    path: str
        Path or name of an AUX_INS/AUX_ITC product, with or without
        .SAFE/.SAFE.zip

    Returns
    -------
    dict
        platform, type ('INS' or 'ITC'), validity and generation (datetime),
        and name (no .SAFE)
    '''
    name = os.path.basename(path)
    for suffix in ('.zip', '.SAFE'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    match = _AUX_NAME.match(name)
    if match is None:
        raise ValueError(f'Not an S1 AUX_INS/AUX_ITC product name: {path}')
    parts = match.groupdict()
    for key in ('validity', 'generation'):
        parts[key] = _parse_compact_time(parts[key])
    parts['name'] = name
    return parts


def _parse_compact_time(text):
    return datetime.datetime.strptime(text, '%Y%m%dT%H%M%S')


def _instrument_value(values, what, path):
    '''One instrument-wide value, refusing to average a beam-dependent one'''
    unique = sorted(set(values))
    if len(unique) != 1:
        raise ValueError(
            f'{what} differs between IW channels in {path} ({unique}); a '
            'single instrument-wide correction would be wrong')
    return unique[0]


def read_platform_timing(aux_ins_zip, aux_itc_zip):
    '''
    Read one platform's timing components from its AUX_INS and AUX_ITC

    Parameters
    ----------
    aux_ins_zip, aux_itc_zip: str
        Paths to the two .SAFE.zip products of the same platform

    Returns
    -------
    dict
        azimuthTimeBias, azimuthCalibration, swstBias, rangeCalibration [s],
        each the same across all IW swath/polarisation channels
    '''
    # Imported here: the SAS only reads tables, and s1reader's aux readers are
    # needed only to build one
    from s1reader.s1_annotation import AuxIns, AuxItc

    ins_platform = parse_aux_name(aux_ins_zip)['platform']
    itc_platform = parse_aux_name(aux_itc_zip)['platform']
    if ins_platform != itc_platform:
        raise ValueError(f'AUX_INS is {ins_platform} but AUX_ITC is '
                         f'{itc_platform}')

    ins = [AuxIns.load_from_zip_file(aux_ins_zip, pol, swath)
           for swath, pol in _IW_CHANNELS]
    itc = [AuxItc.load_from_zip_file(aux_itc_zip, pol, swath)
           for swath, pol in _IW_CHANNELS]
    return {
        'azimuthTimeBias': _instrument_value(
            [a.azimuth_time_bias for a in ins], 'azimuthTimeBias', aux_ins_zip),
        'azimuthCalibration': _instrument_value(
            [a.azimuth_time_total for a in itc], 'azimuth calibration',
            aux_itc_zip),
        'swstBias': _instrument_value(
            [a.swst_bias for a in ins], 'swstBias', aux_ins_zip),
        'rangeCalibration': _instrument_value(
            [a.range_time_total for a in itc], 'range calibration',
            aux_itc_zip),
    }


def _azimuth_total(components):
    return components['azimuthTimeBias'] + components['azimuthCalibration']


def _range_total(components, range_components):
    return sum(components[c] for c in range_components)


def _relative(total, reference_total):
    '''Correction -(T - T_ref); + 0.0 so equal totals write 0.0, not -0.0'''
    return -(total - reference_total) + 0.0


def _group_aux_files(aux_files):
    '''{platform: {'INS': [paths], 'ITC': path}}: one AUX_ITC per platform,
    any number of AUX_INS generations'''
    grouped = {}
    for path in aux_files:
        parts = parse_aux_name(path)
        slot = grouped.setdefault(parts['platform'], {'INS': []})
        if parts['type'] == 'INS':
            if any(parse_aux_name(p)['name'] == parts['name']
                   for p in slot['INS']):
                raise ValueError(f'{parts["name"]} given twice')
            slot['INS'].append(path)
        elif 'ITC' in slot:
            raise ValueError(
                f'Two AUX_ITC products for {parts["platform"]}: '
                f'{slot["ITC"]} and {path}. Pass exactly one.')
        else:
            slot['ITC'] = path
    return grouped


def build_timing_table(aux_files, reference_platform, apply_platforms,
                       generation, range_components=DEFAULT_RANGE_COMPONENTS,
                       validity=None, swath_residuals=None,
                       swath_residual_source=None):
    '''
    Build a relative timing table from ESA aux products

    Parameters
    ----------
    aux_files: list of str
        AUX_INS and AUX_ITC .SAFE.zip paths. One AUX_ITC per platform. For
        the reference platform exactly one AUX_INS: the generation its
        products carry. For a corrected platform every AUX_INS generation its
        products may have been focused with; generations with identical
        values share an entry
    reference_platform: str
        Platform the corrections are relative to, e.g. 'S1C'
    apply_platforms: list of str
        Platforms to correct, e.g. ['S1D']; every other platform gets none
    generation: datetime.datetime
        Generation time of the table (UTC); part of its name
    range_components: tuple of str
        Which of RANGE_COMPONENTS to sum into the range correction
    validity: dict, optional
        {platform: (start, stop)} sensing-time windows; missing or None ends
        stay open
    swath_residuals: dict, optional
        {platform: {'IW1': {'azimuth': s, 'range': s}, ...}}: measured
        per-swath residuals [s] added on top of the aux-derived correction,
        attached to every entry of that platform
    swath_residual_source: str, optional
        Where the residuals came from; required with swath_residuals

    Returns
    -------
    dict
        The table as JSON-ready data; write it with ``write_timing_table``
    '''
    range_components = tuple(range_components)
    unknown = set(range_components) - set(RANGE_COMPONENTS)
    if unknown or not range_components:
        raise ValueError(f'range_components must be a non-empty subset of '
                         f'{RANGE_COMPONENTS}, got {range_components}')
    if reference_platform in apply_platforms:
        raise ValueError(f'{reference_platform} is the reference; it cannot '
                         'also be corrected')
    if not apply_platforms:
        raise ValueError('apply_platforms is empty: nothing to correct')
    validity = validity or {}
    swath_residuals = swath_residuals or {}
    if swath_residuals:
        if not swath_residual_source:
            raise ValueError('swath_residuals need a swath_residual_source')
        unknown = set(swath_residuals) - set(apply_platforms)
        if unknown:
            raise ValueError(f'swath_residuals for {sorted(unknown)}, which '
                             'are not corrected')
        for platform, residuals in swath_residuals.items():
            _check_swath_residuals(residuals, f'swath_residuals[{platform}]')

    grouped = _group_aux_files(aux_files)
    for platform in [reference_platform, *apply_platforms]:
        slot = grouped.get(platform, {'INS': []})
        missing = [t for t in ('INS', 'ITC') if not slot.get(t)]
        if missing:
            raise ValueError(f'No AUX_{"/AUX_".join(missing)} given for '
                             f'{platform}')
    if len(grouped[reference_platform]['INS']) != 1:
        raise ValueError(
            f'Give exactly one AUX_INS for the reference {reference_platform}: '
            'the generation its products carry')

    reference_ins = grouped[reference_platform]['INS'][0]
    reference = read_platform_timing(reference_ins,
                                     grouped[reference_platform]['ITC'])
    entries = []
    for platform in apply_platforms:
        itc = grouped[platform]['ITC']
        start, stop = validity.get(platform) or (None, None)
        # One entry per distinct set of values, covering every generation
        # that carries it, in order of first generation
        groups = {}
        for ins in sorted(grouped[platform]['INS'],
                          key=lambda p: parse_aux_name(p)['generation']):
            comps = read_platform_timing(ins, itc)
            key = tuple(sorted(comps.items()))
            groups.setdefault(key, (comps, []))[1].append(
                parse_aux_name(ins)['name'])
        for comps, aux_ins in groups.values():
            extra = {}
            if platform in swath_residuals:
                extra = {'swathResiduals': swath_residuals[platform],
                         'swathResidualSource': swath_residual_source}
            entries.append({
                'platform': platform,
                'validityStart': _format_time(start),
                'validityStop': _format_time(stop),
                'azimuthTimeCorrection': _relative(
                    _azimuth_total(comps), _azimuth_total(reference)),
                'rangeTimeCorrection': _relative(
                    _range_total(comps, range_components),
                    _range_total(reference, range_components)),
                'rangeComponents': list(range_components),
                'auxIns': aux_ins,
                'auxItc': parse_aux_name(itc)['name'],
                'components': {**comps, 'reference': reference},
                **extra,
            })

    return {
        'productType': PRODUCT_TYPE,
        'name': timing_table_name(reference_platform, generation),
        'generation': _format_time(generation),
        'referencePlatform': reference_platform,
        'referenceAux': {
            'auxIns': parse_aux_name(reference_ins)['name'],
            'auxItc': parse_aux_name(grouped[reference_platform]['ITC'])['name'],
        },
        'convention': CONVENTION,
        'entries': entries,
    }


def timing_table_name(reference_platform, generation):
    '''File stem of a table, e.g. OPERA_S1_AUX_TIMING_S1C-REF_G20261024T000000'''
    return (f'{PRODUCT_TYPE}_{reference_platform}-REF_'
            f'G{generation:%Y%m%dT%H%M%S}')


def _format_time(value):
    return None if value is None else value.strftime('%Y-%m-%dT%H:%M:%SZ')


def _parse_time(value, what):
    if value is None:
        return None
    try:
        return datetime.datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')
    except (TypeError, ValueError):
        raise ValueError(f'{what} must be YYYY-MM-DDTHH:MM:SSZ or null, got '
                         f'{value!r}')


def write_timing_table(table, out_dir):
    '''
    Write a table built by ``build_timing_table`` as <name>.json

    Returns
    -------
    str
        Path written
    '''
    path = os.path.join(out_dir, f'{table["name"]}.json')
    with open(path, 'w') as f:
        json.dump(table, f, indent=2)
        f.write('\n')
    return path


_REQUIRED_ENTRY_KEYS = ('platform', 'validityStart', 'validityStop',
                        'azimuthTimeCorrection', 'rangeTimeCorrection',
                        'rangeComponents', 'auxIns', 'auxItc')


def _check_swath_residuals(residuals, where):
    '''Raise ValueError unless residuals is {swath: {axis: seconds}}'''
    if not isinstance(residuals, dict) or not residuals:
        raise ValueError(f'{where} must be a non-empty map of swaths')
    for swath, values in residuals.items():
        if swath not in SWATHS:
            raise ValueError(f'{where}: swath must be one of {SWATHS}, got '
                             f'{swath!r}')
        if not isinstance(values, dict) or not values:
            raise ValueError(f'{where}.{swath} must map azimuth/range to '
                             'seconds')
        for axis, value in values.items():
            if axis not in _RESIDUAL_AXES:
                raise ValueError(f'{where}.{swath}: axis must be one of '
                                 f'{_RESIDUAL_AXES}, got {axis!r}')
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'{where}.{swath}.{axis} must be a number '
                                 'in seconds')


def _entry_from_json(raw, i):
    where = f'entries[{i}]'
    missing = [k for k in _REQUIRED_ENTRY_KEYS if k not in raw]
    if missing:
        raise ValueError(f'{where} lacks {", ".join(missing)}')
    if not re.fullmatch(r'S1[A-Z]', str(raw['platform'])):
        raise ValueError(f'{where}.platform must look like S1D, got '
                         f'{raw["platform"]!r}')
    for key in ('azimuthTimeCorrection', 'rangeTimeCorrection'):
        if isinstance(raw[key], bool) or not isinstance(raw[key], (int, float)):
            raise ValueError(f'{where}.{key} must be a number in seconds')
    aux_ins = raw['auxIns']
    if (not isinstance(aux_ins, list) or not aux_ins
            or not all(isinstance(n, str) for n in aux_ins)):
        raise ValueError(f'{where}.auxIns must be a non-empty list of AUX_INS '
                         'names')
    range_components = tuple(raw['rangeComponents'])
    if not set(range_components) <= set(RANGE_COMPONENTS):
        raise ValueError(f'{where}.rangeComponents must be a subset of '
                         f'{RANGE_COMPONENTS}')
    residuals = raw.get('swathResiduals') or {}
    source = raw.get('swathResidualSource')
    if residuals:
        _check_swath_residuals(residuals, f'{where}.swathResiduals')
        if not isinstance(source, str) or not source:
            raise ValueError(f'{where}.swathResiduals need a '
                             'swathResidualSource')

    entry = TimingEntry(
        platform=raw['platform'],
        validity_start=_parse_time(raw['validityStart'],
                                   f'{where}.validityStart'),
        validity_stop=_parse_time(raw['validityStop'], f'{where}.validityStop'),
        azimuth_time_correction=float(raw['azimuthTimeCorrection']),
        range_time_correction=float(raw['rangeTimeCorrection']),
        aux_ins=tuple(aux_ins),
        aux_itc=raw['auxItc'],
        components=dict(raw.get('components', {})),
        range_components=range_components,
        swath_residuals=dict(residuals),
        swath_residual_source=source if residuals else None,
    )
    if (entry.validity_start is not None and entry.validity_stop is not None
            and entry.validity_stop <= entry.validity_start):
        raise ValueError(f'{where}: validityStop is not after validityStart')
    return entry


def _windows_overlap(a, b):
    lo = datetime.datetime.min
    hi = datetime.datetime.max
    return ((a.validity_start or lo) < (b.validity_stop or hi)
            and (b.validity_start or lo) < (a.validity_stop or hi))


def load_timing_table(path):
    '''
    Read and validate a relative timing table

    Parameters
    ----------
    path: str
        Path to an OPERA_S1_AUX_TIMING .json file

    Returns
    -------
    TimingTable

    Raises
    ------
    ValueError
        If the file is not a valid table: wrong productType, missing or
        mistyped fields, the reference platform listed as corrected, or one
        AUX_INS generation claimed by two entries whose validity overlaps
    '''
    with open(path) as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as err:
            raise ValueError(f'{path} is not JSON: {err}')

    if raw.get('productType') != PRODUCT_TYPE:
        raise ValueError(f'{path}: productType must be {PRODUCT_TYPE}, got '
                         f'{raw.get("productType")!r}')
    for key in ('name', 'generation', 'referencePlatform', 'entries'):
        if key not in raw:
            raise ValueError(f'{path} lacks {key}')
    if not isinstance(raw['entries'], list):
        raise ValueError(f'{path}: entries must be a list')

    entries = tuple(_entry_from_json(e, i) for i, e in enumerate(raw['entries']))
    reference = raw['referencePlatform']
    for entry in entries:
        if entry.platform == reference:
            raise ValueError(f'{path}: {reference} is the reference platform '
                             'and cannot have an entry')
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            shared = set(a.aux_ins) & set(b.aux_ins)
            if a.platform == b.platform and shared and _windows_overlap(a, b):
                raise ValueError(
                    f'{path}: two {a.platform} entries claim '
                    f'{", ".join(sorted(shared))} with overlapping validity')

    return TimingTable(
        name=raw['name'],
        generation=_parse_time(raw['generation'], 'generation'),
        reference_platform=reference,
        entries=entries,
    )


@dataclass(frozen=True)
class BurstTiming:
    '''
    The timing correction decided for one burst

    Attributes
    ----------
    azimuth_mode: str
        One of AZIMUTH_MODES
    azimuth_offset: float
        Azimuth time offset [s]; 0 unless azimuth_mode is not 'none'
    range_mode: str
        One of RANGE_MODES
    range_offset: float
        Two-way range time offset [s]; 0 unless range_mode is not 'none'
    source: str
        'none', 'constant' or 'table'
    table_name: str or None
        Name of the table the offsets came from
    reference_platform: str or None
        Platform the table's corrections are relative to
    status: str
        Why the offsets are what they are, for the product metadata
    azimuth_swath_residual, range_swath_residual: float
        The part of azimuth_offset / range_offset [s] that came from the
        table's per-swath residual rather than the aux files
    '''
    azimuth_mode: str = 'none'
    azimuth_offset: float = 0.0
    range_mode: str = 'none'
    range_offset: float = 0.0
    source: str = 'none'
    table_name: str | None = None
    reference_platform: str | None = None
    status: str = 'not requested'
    azimuth_swath_residual: float = 0.0
    range_swath_residual: float = 0.0

    @property
    def geometry_azimuth_offset(self):
        '''Offset [s] for the azimuth correction LUT (geometry mode only)'''
        return self.azimuth_offset if self.azimuth_mode == 'geometry' else 0.0

    @property
    def carrier_phase_azimuth_offset(self):
        '''Offset [s] for the azimuth carrier (carrier_phase mode only)'''
        return (self.azimuth_offset if self.azimuth_mode == 'carrier_phase'
                else 0.0)

    @property
    def geometry_range_offset_meters(self):
        '''Slant-range offset [m] for the range correction LUT, c * dt / 2'''
        if self.range_mode != 'geometry':
            return 0.0
        return _SPEED_OF_LIGHT * self.range_offset / 2.0


def is_timing_requested(timing_params):
    '''True if either axis asks for a correction'''
    return (timing_params.azimuth_mode != 'none'
            or timing_params.range_mode != 'none')


def resolve_burst_timing(burst, timing_params, table=None):
    '''
    Decide the timing correction for one burst

    Parameters
    ----------
    burst: Sentinel1BurstSlc
        Burst to correct; platform_id, sensing_start, aux_ins_name and
        swath_name are read
    timing_params: types.SimpleNamespace
        processing.timing_correction
    table: TimingTable or None
        Required when offset_source is 'table' and a mode is requested

    Returns
    -------
    BurstTiming

    Raises
    ------
    UncoveredBurstError
        The table corrects the burst's platform but not this burst, and
        on_uncovered is 'error'
    '''
    if not is_timing_requested(timing_params):
        return BurstTiming()

    az_mode = timing_params.azimuth_mode
    rg_mode = timing_params.range_mode

    def pick(az_value, rg_value, az_residual=0.0, rg_residual=0.0, **kwargs):
        az_on, rg_on = az_mode != 'none', rg_mode != 'none'
        return BurstTiming(
            azimuth_mode=az_mode,
            azimuth_offset=float(az_value) + az_residual if az_on else 0.0,
            range_mode=rg_mode,
            range_offset=float(rg_value) + rg_residual if rg_on else 0.0,
            azimuth_swath_residual=az_residual if az_on else 0.0,
            range_swath_residual=rg_residual if rg_on else 0.0,
            **kwargs)

    if timing_params.offset_source == 'constant':
        return pick(timing_params.azimuth_constant_offset,
                    timing_params.range_constant_offset,
                    source='constant', status='applied')

    if table is None:
        raise ValueError('offset_source is table but no timing table was '
                         'loaded')
    common = dict(source='table', table_name=table.name,
                  reference_platform=table.reference_platform)
    try:
        entry = table.lookup(burst.platform_id, burst.sensing_start,
                             burst.aux_ins_name)
    except UncoveredBurstError as err:
        if timing_params.on_uncovered == 'error':
            raise
        journal.warning('timing_correction.resolve_burst_timing').log(
            f'{err}; no timing correction applied')
        return pick(0.0, 0.0, status=f'uncovered, not applied: {err}',
                    **common)

    if entry is None:
        return pick(0.0, 0.0, status=f'{burst.platform_id} not corrected by '
                                     'the table', **common)
    az_residual, rg_residual = entry.residual(burst.swath_name)
    return pick(entry.azimuth_time_correction, entry.range_time_correction,
                az_residual, rg_residual, status='applied', **common)


def _platform_window(text):
    '''PLATFORM=START[/STOP] -> (platform, (start, stop))'''
    platform, _, window = text.partition('=')
    if not window:
        raise argparse.ArgumentTypeError(
            f'expected PLATFORM=START[/STOP], got {text!r}')
    start, _, stop = window.partition('/')
    parse = (lambda s: datetime.datetime.fromisoformat(s) if s else None)
    try:
        return platform, (parse(start), parse(stop))
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err))


def get_parser():
    '''Command line for building a table'''
    parser = argparse.ArgumentParser(
        description='Build an OPERA_S1_AUX_TIMING table of per-platform '
                    'timing corrections relative to a reference platform, '
                    'from ESA AUX_INS and AUX_ITC products.')
    parser.add_argument('aux_files', nargs='+',
                        help='AUX_INS and AUX_ITC .SAFE.zip files: one AUX_ITC '
                             'per platform, one AUX_INS for the reference, and '
                             'every AUX_INS generation of a corrected platform '
                             '(s1_download_aux --type AUX_INS --platform S1D '
                             '--all)')
    parser.add_argument('--reference', required=True,
                        help='Reference platform, e.g. S1C')
    parser.add_argument('--apply', nargs='+', required=True,
                        help='Platforms to correct, e.g. S1D')
    parser.add_argument('--range-components', nargs='+',
                        default=list(DEFAULT_RANGE_COMPONENTS),
                        choices=RANGE_COMPONENTS,
                        help='Components summed into the range correction '
                             '(default: rangeCalibration only)')
    parser.add_argument('--validity', type=_platform_window, action='append',
                        default=[], metavar='PLATFORM=START[/STOP]',
                        help='Sensing-time window for a platform, ISO dates; '
                             'open-ended by default')
    parser.add_argument('--swath-residuals', default=None, metavar='JSON',
                        help='Measured per-swath residuals added on top of the '
                             'aux-derived correction: {"source": "...", '
                             '"S1D": {"IW1": {"azimuth": seconds}, ...}}')
    parser.add_argument('--generation', type=datetime.datetime.fromisoformat,
                        default=None,
                        help='Table generation time, ISO (default: now, UTC)')
    parser.add_argument('-o', '--out-dir', default='.',
                        help='Directory to write the table into')
    return parser


def main(argv=None):
    '''Build and write a table; print its path'''
    args = get_parser().parse_args(argv)
    generation = args.generation or datetime.datetime.now(
        datetime.timezone.utc).replace(tzinfo=None, microsecond=0)
    residuals, source = None, None
    if args.swath_residuals:
        with open(args.swath_residuals) as f:
            residuals = json.load(f)
        source = residuals.pop('source', None)
    table = build_timing_table(
        args.aux_files, args.reference, args.apply, generation,
        range_components=args.range_components,
        validity=dict(args.validity), swath_residuals=residuals,
        swath_residual_source=source)
    path = write_timing_table(table, args.out_dir)
    print(path)
    return path


if __name__ == '__main__':
    main()

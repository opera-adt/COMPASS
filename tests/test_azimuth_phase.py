import types

import isce3
import numpy as np
import pytest

from compass.utils.azimuth_phase import apply_azimuth_phase_offset

# A small radar grid in the attribute names isce3's RadarGridParameters uses
GRID = types.SimpleNamespace(sensing_start=100.0, prf=486.0,
                             starting_range=800_000.0,
                             range_pixel_spacing=2.33)
SHAPE = (37, 23)


def _poly(coeffs, y_mean=100.02, x_mean=800_025.0, y_std=0.04, x_std=25.0):
    '''Poly2d in (azimuth time, slant range), as s1-reader fits the carrier.'''
    return isce3.core.Poly2d(np.asarray(coeffs, dtype=float), x_mean, y_mean,
                             x_std, y_std)


P0 = _poly([[0.3, 1.2, -0.4], [2.0, 0.5, 0.1], [-7.0, 0.2, 0.0]])
P1 = _poly([[0.1, 1.1, -0.3], [2.6, 0.4, 0.2], [-6.5, 0.3, 0.05]])


def _data(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=SHAPE) + 1j * rng.normal(size=SHAPE)
            ).astype(np.complex64)


def _isce3_coords():
    '''The coordinates isce3 geocode_slc's carrierPhaseDeramp evaluates at.'''
    lines, samples = np.indices(SHAPE)
    return (GRID.sensing_start + lines / GRID.prf,
            GRID.starting_range + samples * GRID.range_pixel_spacing)


def test_equal_carriers_leave_data_unchanged():
    d = _data()
    out = apply_azimuth_phase_offset(d.copy(), GRID, P0, P0)
    np.testing.assert_array_equal(out, d)


def test_deramp_with_shifted_carrier_recovers_true_baseband():
    '''The promise geocode_slc depends on: deramping the corrected data with
    P1 gives exactly what deramping the original with P0 gives.'''
    d = _data()
    out = apply_azimuth_phase_offset(d.copy(), GRID, P0, P1)
    az, rg = _isce3_coords()
    np.testing.assert_allclose(out * np.exp(-1j * P1.eval(az, rg)),
                               d * np.exp(-1j * P0.eval(az, rg)),
                               rtol=0, atol=1e-5)


def test_amplitude_is_untouched():
    d = _data()
    out = apply_azimuth_phase_offset(d.copy(), GRID, P0, P1)
    np.testing.assert_allclose(np.abs(out), np.abs(d), rtol=1e-6)


def test_modifies_in_place_and_keeps_dtype():
    d = _data()
    out = apply_azimuth_phase_offset(d, GRID, P0, P1)
    assert out is d
    assert out.dtype == np.complex64


@pytest.mark.parametrize("block_lines", [1, 5, 36, 37, 1000])
def test_result_does_not_depend_on_block_size(block_lines):
    ref = apply_azimuth_phase_offset(_data(), GRID, P0, P1, block_lines=256)
    out = apply_azimuth_phase_offset(_data(), GRID, P0, P1,
                                     block_lines=block_lines)
    np.testing.assert_array_equal(out, ref)


def test_screen_varies_along_azimuth_at_isce3_line_times():
    '''A carrier quadratic in azimuth time gives a line-dependent screen; the
    value at line i must be evaluated at sensing_start + i / prf.'''
    q0 = _poly([[0.0], [0.0], [5.0]])          # 5 * ((t - y_mean) / y_std)^2
    q1 = _poly([[0.0], [0.0], [5.0]], y_mean=100.021)
    ones = np.ones(SHAPE, dtype=np.complex64)
    out = apply_azimuth_phase_offset(ones, GRID, q0, q1)
    for line in (0, 11, 36):
        t = GRID.sensing_start + line / GRID.prf
        expected = 5 * ((t - 100.021) / 0.04) ** 2 - 5 * ((t - 100.02) / 0.04) ** 2
        np.testing.assert_allclose(np.angle(out[line, 0]),
                                   np.angle(np.exp(1j * expected)), atol=1e-5)

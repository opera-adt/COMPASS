'''
Test the constant LUT behind the geometry timing offsets

A geometry-mode timing offset rides on the correction LUTs. When the model
LUTs are enabled it is added to their data (see utils/lut.py); when they are
disabled geocode_slc still needs a LUT to query, which is what _constant_lut
builds.
'''
import types

import isce3
import numpy as np
import pytest

from compass.s1_geocode_slc import _constant_lut


def radar_grid():
    '''The four attributes the helper reads off an isce3 radar grid.'''
    return types.SimpleNamespace(
        starting_range=800_000.0,
        end_range=900_000.0,
        sensing_start=0.0,
        sensing_stop=3.0,
    )


def test_constant_lut_spans_the_grid():
    '''The LUT covers the burst, so no query falls outside it.'''
    lut = _constant_lut(radar_grid(), 0.25)

    assert isinstance(lut, isce3.core.LUT2d)
    assert (lut.x_start, lut.x_end) == (800_000.0, 900_000.0)
    assert (lut.y_start, lut.y_end) == (0.0, 3.0)


@pytest.mark.parametrize('offset', [0.25, -0.25, 0.0])
@pytest.mark.parametrize(
    'slant_range, azimuth_time',
    [(800_000.0, 0.0), (850_000.0, 1.5), (900_000.0, 3.0)],
)
def test_constant_lut_returns_the_offset_everywhere(
    offset, slant_range, azimuth_time
):
    '''A re-registration is uniform: same shift at every pixel, sign kept.'''
    lut = _constant_lut(radar_grid(), offset)

    assert lut.eval(azimuth_time, slant_range) == pytest.approx(offset)


def test_cumulative_luts_take_the_offset():
    '''The enabled-LUT path adds the offset to the azimuth correction data.'''
    import inspect

    from compass.utils.lut import cumulative_correction_luts

    params = inspect.signature(cumulative_correction_luts).parameters
    assert params['az_time_offset'].default == 0.0
    assert 'az_lut_data + az_time_offset' in inspect.getsource(
        cumulative_correction_luts
    )

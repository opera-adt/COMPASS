'''
Phase-only azimuth-time correction for TOPS bursts.

An azimuth timing error dt shows up in two places: the geolocation of the
data, and the phase of the TOPS azimuth carrier (and so in the burst-overlap
phase that spectral diversity measures). When the geolocation is already
correct and only the carrier phase carries the error -- the case for the
Sentinel-1 AUX_INS/AUX_ITC timing constants in delivered IPF products -- the
correction must change the phase without resampling the data.

geocode_slc deramps the radar data with a carrier polynomial P, interpolates,
and re-ramps with the same P at each output pixel. If the radar data is first
multiplied by exp(i*(P1 - P0)), where P1(t) = P0(t + dt) is the carrier of the
shifted burst, and geocode_slc is then given P1, the deramp still yields the
true baseband (D*exp(i*(P1-P0))*exp(-i*P1) = D*exp(-i*P0)) while the re-ramp
applies P1. The output differs from an uncorrected product by exactly
exp(i*(P0(t + dt) - P0(t))): the phase part of the timing shift, with sampling
and geometry untouched.
'''
import numpy as np


def apply_azimuth_phase_offset(rdr_data, radar_grid, carrier_ref,
                               carrier_shifted, block_lines=256):
    '''
    Multiply radar-grid SLC data in place by exp(i*(P1 - P0)).

    Both carriers are evaluated at the coordinates isce3's geocode_slc uses
    for its own deramp: azimuth time ``sensing_start + line / prf`` and slant
    range ``starting_range + sample * range_pixel_spacing`` over the full
    radar grid, so the pre-multiplication cancels exactly against the deramp
    with ``carrier_shifted``.

    Parameters
    ----------
    rdr_data: np.ndarray
        Complex radar-grid SLC block, shape (lines, samples), modified in place
    radar_grid: isce3.product.RadarGridParameters
        Radar grid the block spans (the grid passed to geocode_slc)
    carrier_ref: isce3.core.Poly2d
        Azimuth carrier polynomial P0 of the burst, as (azimuth time, range)
    carrier_shifted: isce3.core.Poly2d
        Azimuth carrier P1 of the burst shifted by the phase offset, i.e. what
        ``burst.get_az_carrier_poly(offset=dt / burst.azimuth_time_interval)``
        returns
    block_lines: int
        Azimuth lines evaluated at a time, to bound memory

    Returns
    -------
    np.ndarray
        ``rdr_data``, for convenience
    '''
    n_lines, n_samples = rdr_data.shape
    rng = radar_grid.starting_range + \
        np.arange(n_samples) * radar_grid.range_pixel_spacing
    for i0 in range(0, n_lines, block_lines):
        i1 = min(n_lines, i0 + block_lines)
        az = radar_grid.sensing_start + np.arange(i0, i1) / radar_grid.prf
        az_mesh, rng_mesh = np.meshgrid(az, rng, indexing='ij')
        dphi = carrier_shifted.eval(az_mesh, rng_mesh) - \
            carrier_ref.eval(az_mesh, rng_mesh)
        rdr_data[i0:i1] *= np.exp(1j * dphi).astype(rdr_data.dtype)
    return rdr_data

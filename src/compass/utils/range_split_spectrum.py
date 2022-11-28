import isce3
import numpy as np
from compass.utils.elevation_antenna_pattern import apply_eap_correction
from isce3.splitspectrum import splitspectrum
from osgeo import gdal

def find_next_power(number):
    '''
    Finds the next power of 2 of 'number'

    Parameters
    ----------
    number: int
        Number for which to find next power of two

    Returns
    -------
    power: int
        Next power of 2 of 'number'
    '''
    power = 1
    if (number and not (number & (number - 1))):
        return number

    while (power < number):
        power <<= 1
    return number


def range_split_spectrum(burst,
                         burst_path,
                         cfg_split_spectrum,
                         scratch_path):
    '''
    Split burst range spectrum
    Parameters
    ----------
    burst: Sentinel1BurstSlc
        S1-A/B burst object
    burst_path: str
        Path to the burst SLC to apply split spectrum
    cfg_split_spectrum: dict
        Dictionary with split-spetrum options
    scratch_path: str
        Directory for storing temp files

    Returns
    -------
    burst_raster: isce3.io.Raster
        3-bands ISCE3 Raster. Band #1: low band;
        Band #2: main band; Band #3: high band
    '''
    length, width = burst.shape
    lines_per_block = cfg_split_spectrum.lines_per_block
    burst_id_pol = f'{burst.burst_id}_{burst.polarization}'

    # In ISCE3, we can use raised cosine to implement S1-A/B Hamming
    window_type = burst.range_window_type
    window_type = 'Cosine' if window_type.casefold() == 'hamming' else window_type
    window_shape = 2 * burst.range_window_coefficient - 1.0 if \
    window_type.casefold() == 'hamming' else burst.range_window_coefficient

    # Extract bandwidths (bw) and create frequency vectors
    half_bw = 0.5 * burst.range_bandwidth
    low_bw = cfg_split_spectrum.low_band_bandwidth
    high_bw = cfg_split_spectrum.high_band_bandwidth

    low_freq_burst = burst.radar_center_frequency - half_bw
    high_freq_burst = burst.radar_center_frequency + half_bw

    low_band_freqs = np.array([low_freq_burst, low_freq_burst + low_bw])
    high_band_freqs = np.array([high_freq_burst - high_bw, high_freq_burst])
    low_center_freq = low_freq_burst + low_bw / 2
    high_center_freq = high_freq_burst - high_bw / 2

    # Initialize the split-spectrum parameter object. Note, constructor
    # requires frequency (A/B) but this is not used anywhere below
    rdr_grid = burst.as_isce3_radargrid()
    split_spectrum_obj = splitspectrum.SplitSpectrum(
        rg_sample_freq=burst.range_sampling_rate,
        rg_bandwidth=burst.range_bandwidth,
        center_frequency=burst.radar_center_frequency,
        slant_range=rdr_grid.slant_range,
        freq='A')


    # The output burst will
    # contain 3 bands: Band #1: low-band image; Band #2 main-band image;
    # Band #3: high-band image.
    in_ds = gdal.Open(burst_path, gdal.GA_ReadOnly)
    driver = gdal.GetDriverByName('ENVI')
    out_ds = driver.Create(f'{scratch_path}/{burst_id_pol}_low_main_high.slc',
                           width, length, 3, gdal.GDT_CFloat32)

    # Prepare necessary variables for block processing
    lines_per_block = min(length, lines_per_block)
    num_blocks = int(np.ceil(length / lines_per_block))

    for block in range(num_blocks):
        line_start = block * lines_per_block
        if block == num_blocks - 1:
            block_length = length - line_start
        else:
            block_length = lines_per_block

        # Read a block of valid burst data
        burst_data = in_ds.GetRasterBand(1).ReadAsArray(0, line_start,
                                                        width, block_length)
        # Get the low band sub-image and corresponding metadata
        burst_low_data, _ = split_spectrum_obj.bandpass_shift_spectrum(
            slc_raster=burst_data, low_frequency=low_band_freqs[0],
            high_frequency=low_band_freqs[1],
            new_center_frequency=low_center_freq,
            fft_size=find_next_power(width), window_shape=window_shape,
            window_function=window_type, resampling=False
        )
        # Get the high sub-image and corresponding metadata
        burst_high_data, _ = split_spectrum_obj.bandpass_shift_spectrum(
            slc_raster=burst_data, low_frequency=high_band_freqs[0],
            high_frequency=high_band_freqs[1],
            new_center_frequency=high_center_freq,
            fft_size=find_next_power(width), window_shape=window_shape,
            window_function=window_type, resampling=False
        )
        # Write back all the processed data
        out_ds.GetRasterBand(1).WriteArray(burst_low_data[0:block_length],
                                           yoff=line_start)
        out_ds.GetRasterBand(2).WriteArray(burst_data[0:block_length],
                                           yoff=line_start)
        out_ds.GetRasterBand(3).WriteArray(burst_high_data[0:block_length],
                                           yoff=line_start)

    out_ds.FlushCache()
    out_ds = None
    burst_raster = isce3.io.Raster(
        f'{scratch_path}/{burst_id_pol}_low_main_high.slc')

    return burst_raster

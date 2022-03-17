import isce3
import numpy as np
from isce3.splitspectrum import splitspectrum
from osgeo import gdal


def save_valid_burst(burst, lines_per_block,
                     scratch_path):
    '''
    Save locally the valid part of a S1-A/B burst
    (Done with block processing to avoid memory issues)

    Parameters:
    ----------
    burst: Sentinel1BurstSlc
       S1-A/B burst object
    lines_per_block: int
       Number of lines to read/process in batch
    scratch_path: str
       Directory where to store temp data

    Returns:
    -------
    burst_path: str
       Filepath to the locally
    '''
    valid_length = burst.last_valid_line - burst.first_valid_line
    valid_width = burst.last_valid_sample - burst.first_valid_sample

    invalid_burst_path = f'{scratch_path}/invalid_burst_temp'
    valid_burst_path = invalid_burst_path.replace('invalid', 'valid')

    burst.slc_to_file(invalid_burst_path)
    # Read invalid burst and create output valid burst
    in_ds = gdal.Open(invalid_burst_path, gdal.GA_ReadOnly)
    driver = gdal.GetDriverByName('ENVI')
    out_ds = driver.Create(valid_burst_path, valid_width, valid_length,
                           in_ds.RasterCount, gdal.GDT_CFloat32)

    # Start block processing
    lines_per_block = min(valid_length, lines_per_block)
    num_blocks = int(np.ceil(valid_length / lines_per_block))

    for band in range(in_ds.RasterCount):
        for block in range(num_blocks):
            line_start = block * lines_per_block
            if block == num_blocks - 1:
                block_length = valid_length - line_start
            else:
                block_length = lines_per_block

            # Read a block of burst data
            burst_data = in_ds.GetRasterBand(band+1).ReadAsArray(
                burst.first_valid_sample,
                burst.first_valid_line + line_start,
                valid_width, block_length)

            # Write extracted data
            out_ds.GetRasterBand(band+1).WriteArray(burst_data, yoff=line_start)
    out_ds.FlushCache()
    return valid_burst_path


def range_split_spectrum(burst, cfg_split_spectrum,
                         scratch_path):
    '''
    Split burst range spectrum
    Parameters:
    ----------
    burst: Sentinel1BurstSlc
       S1-A/B burst object
    cfg_split_spectrum: dict
       Dictionary with split-spetrum options
    scratch_path: str
       Directory for storing temp files

    Returns:
    -------
    burst_raster: isce3.io.Raster
       3-bands ISCE3 Raster. Band #1: low band;
       Band #2: main band; Band #3: high band
    '''
    length, width = burst.shape
    lines_per_block = cfg_split_spectrum.lines_per_block

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
    split_spectrum_params = splitspectrum.SplitSpectrum(
        rg_sample_freq=burst.range_sampling_rate,
        rg_bandwidth=burst.range_bandwidth,
        center_frequency=burst.radar_center_frequency,
        slant_range=rdr_grid.slant_range,
        freq='A')

    # Save locally the valid part of the burst. Note, invalid data (zeros)
    # cause issue when computing fft when splitting the range bandwidth
    valid_burst_path = save_valid_burst(burst, lines_per_block, scratch_path)

    # Open valid burst and create output burst. The output burst will
    # contain 3 bands: Band #1: low-band image; Band #2 main-band image;
    # Band #3: high-band image.
    in_ds = gdal.Open(valid_burst_path, gdal.GA_ReadOnly)
    valid_length = in_ds.RasterYSize
    valid_width = in_ds.RasterXSize
    driver = gdal.GetDriverByName('ENVI')
    out_ds = driver.Create(f'{scratch_path}/{burst.burst_id}_low_main_high',
                           width, length, 3, gdal.GDT_CFloat32)

    # Prepare necessary variables for block processing
    lines_per_block = min(valid_length, lines_per_block)
    num_blocks = int(np.ceil(valid_length / lines_per_block))

    for block in range(num_blocks):
        line_start = block * lines_per_block
        if block == num_blocks - 1:
            block_length = valid_length - line_start
        else:
            block_length = lines_per_block

        # Read a block of valid burst data
        burst_data = in_ds.GetRasterBand(1).ReadAsArray(0, line_start,
                                                        valid_width, block_length)
        # Get the low band sub-image and corresponding metadata
        burst_low_data, burst_low_meta = split_spectrum_params.bandpass_shift_spectrum(
            slc_raster=burst_data, low_frequency=low_band_freqs[0],
            high_frequency=low_band_freqs[1],
            new_center_frequency=low_center_freq,
            fft_size=valid_width, window_shape=burst.range_window_coefficient,
            window_function=burst.range_window_type, resampling=False
        )
        # Get the high sub-image and corresponding metadata
        burst_high_data, burst_high_metadata = split_spectrum_params.bandpass_shift_spectrum(
            slc_raster=burst_data, low_frequency=high_band_freqs[0],
            high_frequency=high_band_freqs[1],
            new_center_frequency=high_center_freq,
            fft_size=valid_width, window_shape=burst.range_window_coefficient,
            window_function=burst.range_window_type, resampling=False
        )
        # Write back all the processed data
        out_ds.GetRasterBand(1).WriteArray(burst_low_data[0:block_length],
                                           yoff=line_start + burst.first_valid_line,
                                           xoff=burst.first_valid_sample)
        out_ds.GetRasterBand(2).WriteArray(burst_data[0:block_length],
                                           yoff=line_start + burst.first_valid_line,
                                           xoff=burst.first_valid_sample)
        out_ds.GetRasterBand(3).WriteArray(burst_high_data[0:block_length],
                                           yoff=line_start + burst.first_valid_line,
                                           xoff=burst.first_valid_sample)

    out_ds.FlushCache()
    out_ds = None
    burst_raster = isce3.io.Raster(
        f'{scratch_path}/{burst.burst_id}_low_main_high')

    return burst_raster

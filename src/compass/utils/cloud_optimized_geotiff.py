import os
import shutil
import tempfile
import journal
from osgeo import gdal, osr
from compass.utils import validate_cloud_optimized_geotiff.main as validate_cog

def save_as_cog(filename, scratch_dir = '.', flag_compress=True,
                resamp_algorithm=None):
    """Save (overwrite) a GeoTIFF file as a cloud-optimized GeoTIFF.
       Parameters
       ----------
       filename: str
              GeoTIFF to be saved as a cloud-optimized GeoTIFF
       scratch_dir: str (optional)
              Temporary Directory
       flag_compress: bool (optional)
              Flag to indicate whether images should be
              compressed
       resamp_algorithm: str (optional)
              Resampling algorithm. Options: "AVERAGE",
              "AVERAGE_MAGPHASE", "RMS", "BILINEAR",
              "CUBIC", "CUBICSPLINE", "GAUSS", "LANCZOS",
              "MODE", "NEAREST", or "NONE"
    """
    warning_channel = journal.warning('cloud_optimized_geotiff.save_as_cog')
    info_channel = journal.info('cloud_optimized_geotiff.save_as_cog')

    info_channel.log('COG step 1: add overviews')
    gdal_ds = gdal.Open(filename, 1)
    gdal_dtype = gdal_ds.GetRasterBand(1).DataType
    dtype_name = gdal.GetDataTypeName(gdal_dtype).lower()
    is_integer = 'byte' in dtype_name  or 'int' in dtype_name

    overviews_list = [4, 16, 64, 128]

    if is_integer:
        resamp_algorithm = 'NEAREST'
    else:
        resamp_algorithm = 'CUBICSPLINE'

    gdal_ds.BuildOverviews('CUBICSPLINE', overviews_list,
                           gdal.TermProgress_nocb)

    del gdal_ds  # close the dataset (Python object and pointers)
    external_overview_file = filename + '.ovr'
    if os.path.isfile(external_overview_file):
        os.path.remove(external_overview_file)

    info_channel.log('COG step 2: save as COG')
    temp_file = tempfile.NamedTemporaryFile(
                    dir=scratch_dir, suffix='.tif').name

    tile_size = 256
    ovr_tile_size = tile_size
    gdal_translate_options = [
        'TILED=YES',
        f'BLOCKXSIZE={tile_size}',
        f'BLOCKYSIZE={tile_size}',
        f'GDAL_TIFF_OVR_BLOCKSIZE={ovr_tile_size}'
        'COPY_SRC_OVERVIEWS=YES']

    if flag_compress:
        gdal_translate_options += ['COMPRESS=DEFLATE']

    if is_integer:
        gdal_translate_options += ['PREDICTOR=2']
    else:
        gdal_translate_options = ['PREDICTOR=3']

    gdal.Translate(temp_file, filename,
                   creationOptions=gdal_translate_options)

    shutil.move(temp_file, filename)

    info_channel.log('COG step 3: validate')

    argv = ['--full-check=yes', filename]
    validate_cog_ret = validate_cog(argv)
    if validate_cog_ret == 0:
        info_channel.log(f'file "{filename}" is a valid cloud optimized'
                         ' GeoTIFF')
    else:
        warning_channel.log(f'file "{filename}" is NOT a valid cloud'
                            f' optimized GeoTIFF!')
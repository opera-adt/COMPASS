''' A routine to apply elevation antenna pattern correction
(EAP correction)'''


import numpy as np
from osgeo import gdal


def apply_eap_correction(burst, path_slc_vrt, path_slc_corrected, check_eap):
    '''
    Apply Elevation Antenna Pattern correction (EAP correction)
    on the input burst

    Parameters:
    -----------
    burst: Sentinel1BurstSlc
        Input burst
    path_slc_vrt: str
        Path to the burst SLC to be corrected
    path_slc_corrected: str:
        Path to the burst SLC after EAP correction
    check_eap: Namespace
        A namespace that contains flags if phase and/or magnitude
        EAP correction are necessary

    '''

    # Retrieve the EAP correction in range
    vec_eap_line = burst.eap_compensation_lut

    # Remove magnitude component when we don't need to correct it
    if check_eap.phase_correction and not check_eap.magnitude_correction:
        vec_eap_line /= np.abs(vec_eap_line)

    # Load the burst SLC to correct
    slc_in = gdal.Open(path_slc_vrt, gdal.GA_ReadOnly)
    arr_slc_in = slc_in.ReadAsArray()

    # Shape the correction vector to the size of burst
    array_eap_line = (  vec_eap_line[np.newaxis, ...]
                      * np.ones((arr_slc_in.shape[0], 1)))

    # Apply the correction
    arr_slc_corrected = arr_slc_in / array_eap_line

    # Write out the EAP-corrected SLC
    dtype = slc_in.GetRasterBand(1).DataType
    drvout = gdal.GetDriverByName('GTiff')
    raster_out = drvout.Create(path_slc_corrected, burst.shape[1],
                               burst.shape[0], 1, dtype)
    band_out = raster_out.GetRasterBand(1)
    band_out.WriteArray(arr_slc_corrected)
    band_out.FlushCache()
    del band_out
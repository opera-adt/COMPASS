#!/usr/bin/env python

'''wrapper for geocoded SLC'''

from datetime import timedelta
import os
import time

import h5py
import isce3
import journal
from nisar.workflows.h5_prep import set_get_geo_info
import numpy as np

from compass import s1_rdr2geo
from compass import s1_geocode_metadata
from compass.utils.geo_metadata import GeoCslcMetadata
from compass.utils.geo_runconfig import GeoRunConfig
from compass.utils.helpers import get_module_name
from compass.utils.range_split_spectrum import range_split_spectrum
from compass.utils.yaml_argparse import YamlArgparse


def init_bs_dataset(h5_root, polarization, geo_grid):
    '''
    Create and allocate dataset for isce.geocode.geocode_slc to write to

    h5_root: h5py
        Root to CSLC
    polarization: str
        Polarization to be used as dataset name
    geo_grid: isce3.product.GeoGridParameters
        Geogrid out output
    '''
    bs_group = h5_root.require_group('complex_backscatter')

    # Data type
    ctype = h5py.h5t.py_create(np.complex64)
    ctype.commit(h5_root['/'].id, np.string_('complex64'))

    shape = (geo_grid.length, geo_grid.width)
    bs_ds = bs_group.create_dataset(polarization, dtype=ctype, shape=shape)

    descr = f'Geocoded SLC image ({polarization})'
    bs_ds.attrs['description'] = np.string_(descr)

    long_name = f'geocoded single-look complex image {polarization}'
    bs_ds.attrs['long_name'] = np.string_(long_name)

    yds, xds = set_get_geo_info(h5_root, '/', geo_grid)

    bs_ds.dims[0].attach_scale(yds)
    bs_ds.dims[1].attach_scale(xds)


def run(cfg: GeoRunConfig):
    '''
    Run geocode burst workflow with user-defined
    args stored in dictionary runconfig *cfg*

    Parameters
    ---------
    cfg: GeoRunConfig
        GeoRunConfig object with user runconfig options
    '''
    module_name = get_module_name(__file__)
    info_channel = journal.info(f"{module_name}.run")
    info_channel.log(f"Starting {module_name} burst")

    # Start tracking processing time
    t_start = time.time()

    # Common initializations
    image_grid_doppler = isce3.core.LUT2d()
    threshold = cfg.geo2rdr_params.threshold
    iters = cfg.geo2rdr_params.numiter
    blocksize = cfg.geo2rdr_params.lines_per_block
    flatten = cfg.geocoding_params.flatten

    # process one burst only
    for burst in cfg.bursts:
        # Reinitialize the dem raster per burst to prevent raster artifacts
        # caused by modification in geocodeSlc
        dem_raster = isce3.io.Raster(cfg.dem)
        epsg = dem_raster.get_epsg()
        proj = isce3.core.make_projection(epsg)
        ellipsoid = proj.ellipsoid

        date_str = burst.sensing_start.strftime("%Y%m%d")
        burst_id = burst.burst_id
        pol = burst.polarization
        id_pol = f"{burst_id}_{pol}"
        geo_grid = cfg.geogrids[burst_id]

        scratch_path = f'{cfg.scratch_path}/{burst_id}/{date_str}'
        os.makedirs(scratch_path, exist_ok=True)

        radar_grid = burst.as_isce3_radargrid()
        native_doppler = burst.doppler.lut2d
        orbit = burst.orbit

        # Get azimuth polynomial coefficients for this burst
        az_carrier_poly2d = burst.get_az_carrier_poly()

        # Generate required metadata layers
        # TODO seperate metadata file as needed
        if cfg.rdr2geo_params.enabled:
            s1_rdr2geo.run(cfg, save_in_scratch=True)
            if cfg.rdr2geo_params.geocode_metadata_layers:
                s1_geocode_metadata.run(cfg, fetch_from_scratch=True)

        # Split the range bandwidth of the burst, if required
        if cfg.split_spectrum_params.enabled:
            rdr_burst_raster = range_split_spectrum(burst,
                                                    cfg.split_spectrum_params,
                                                    scratch_path)
        else:
            temp_slc_path = f'{scratch_path}/{id_pol}_temp.vrt'
            burst.slc_to_vrt_file(temp_slc_path)
            rdr_burst_raster = isce3.io.Raster(temp_slc_path)

        # Extract burst boundaries
        b_bounds = np.s_[burst.first_valid_line:burst.last_valid_line,
                   burst.first_valid_sample:burst.last_valid_sample]

        # Create sliced radar grid representing valid region of the burst
        sliced_radar_grid = burst.as_isce3_radargrid()[b_bounds]

        # Create top output path
        burst_output_path = f'{cfg.product_path}/{burst_id}/{date_str}'
        os.makedirs(burst_output_path, exist_ok=True)

        output_hdf5 = f'{burst_output_path}/{id_pol}.hdf5'
        with h5py.File(output_hdf5, 'w') as geo_burst_h5:
            geo_burst_h5.attrs['Conventions'] = np.string_("CF-1.8")
            bs_group = geo_burst_h5.require_group('complex_backscatter')
            init_bs_dataset(geo_burst_h5, pol, geo_grid)

            # access the HDF5 dataset for a given frequency and polarization
            dataset_path = f'/complex_backscatter/{pol}'
            gslc_dataset = geo_burst_h5[dataset_path]

            # Construct the output raster directly from HDF5 dataset
            geo_burst_raster = isce3.io.Raster(f"IH5:::ID={gslc_dataset.id.id}".encode("utf-8"), update=True)

            # Geocode
            isce3.geocode.geocode_slc(geo_burst_raster, rdr_burst_raster,
                                      dem_raster,
                                      radar_grid, sliced_radar_grid,
                                      geo_grid, orbit,
                                      native_doppler,
                                      image_grid_doppler, ellipsoid, threshold,
                                      iters, blocksize, flatten,
                                      azimuth_carrier=az_carrier_poly2d)

            # Set geo transformation
            geotransform = [geo_grid.start_x, geo_grid.spacing_x, 0,
                            geo_grid.start_y, 0, geo_grid.spacing_y]
            geo_burst_raster.set_geotransform(geotransform)
            geo_burst_raster.set_epsg(epsg)
            del geo_burst_raster
            del dem_raster # modified in geocodeSlc

            # Save burst metadata
            metadata = GeoCslcMetadata.from_georunconfig(cfg, burst_id)
            metadata.to_hdf5(geo_burst_h5)
            geo_burst_h5['metadata/runconfig'] = np.string_(cfg.yaml_string)


    dt = str(timedelta(seconds=time.time() - t_start)).split(".")[0]
    info_channel.log(f"{module_name} burst successfully ran in {dt} (hr:min:sec)")


if __name__ == "__main__":
    '''Run geocode cslc workflow from command line'''
    # load arguments from command line
    parser = YamlArgparse()

    # Get a runconfig dict from command line argumens
    cfg = GeoRunConfig.load_from_yaml(parser.run_config_path,
                                      workflow_name='s1_cslc_geo')

    # Run geocode burst workflow
    run(cfg)

import isce3
import journal
import os
import time
from osgeo import gdal

from compass.utils.runconfig import RunConfig
from compass.utils.yaml_argparse import YamlArgparse
from compass.utils.geogrid import generate_geogrid


def run(cfg):
    '''
    Run geocode burst workflow with user-defined
    args stored in dictionary runconfig *cfg*

    Parameter:
    ---------
    cfg: dict
       Dictionary with user runconfig options
    '''
    info_channel = journal.info("geo_cslc.run")

    # Start tracking processing time
    t_start = time.time()
    info_channel.log("Starting geocode burst")

    # Common initializations
    dem_raster = isce3.io.Raster(cfg.dem)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid
    image_grid_doppler = isce3.core.LUT2d()
    threshold = cfg.geo2rdr_params.threshold
    iters = cfg.geo2rdr_params.numiter
    blocksize = cfg.geo2rdr_params.lines_per_block
    dem_margin = cfg.geocoding_params.dem_margin
    flatten = cfg.geocoding_params.flatten

    for bursts in cfg.bursts:
        output_path = f'{cfg.product_path}/{bursts[0].burst_id}'
        scratch_path = f'{cfg.scratch_path}/{bursts[0].burst_id}'
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(scratch_path, exist_ok=True)
        for burst in bursts:
            radar_grid = burst.as_isce3_radargrid()
            native_doppler = burst.doppler.lut2d
            orbit = burst.orbit

            # Generate geogrid for the burst
            geo_grid = generate_geogrid(radar_grid, orbit, dem_raster,
                                        cfg.geocoding_params)
            # Get azimuth polynomial coefficients for this burst
            az_carrier_poly2d = burst.get_az_carrier_poly()

            # Save burst prior to geocoding
            temp_slc_path = f'{scratch_path}/{burst.burst_id}_temp.tiff'
            burst.slc_to_file(temp_slc_path, 'GTiff')
            rdr_burst_raster = isce3.io.Raster(temp_slc_path)

            # Generate output geocoded burst raster
            geo_burst_raster = isce3.io.Raster(
                f'{output_path}/geo_{burst.burst_id}.tiff',
                geo_grid.width, geo_grid.length,
                rdr_burst_raster.num_bands, gdal.GDT_CFloat32,
                'GTiff')
            # Geocode
            isce3.geocode.geocode_slc(geo_burst_raster, rdr_burst_raster,
                                      dem_raster,
                                      radar_grid, geo_grid, orbit,
                                      native_doppler,
                                      image_grid_doppler, ellipsoid, threshold,
                                      iters,
                                      blocksize, dem_margin, flatten,
                                      azimuth_carrier=az_carrier_poly2d)
            # Set geo transformation
            geotransform = [geo_grid.start_x, geo_grid.spacing_x, 0,
                            geo_grid.start_y, 0, geo_grid.spacing_y]
            geo_burst_raster.set_geotransform(geotransform)
            geo_burst_raster.set_epsg(epsg)
    dt = time.time() - t_start
    info_channel.log(f'geocode burst successfully ran in {dt:.3f} seconds')


if __name__ == "__main__":
    '''Run geocode cslc workflow from command line'''
    # load arguments from command line
    geo_parser = YamlArgparse()

    # Get a runconfig dict from command line argumens
    runconfig = RunConfig.load_from_yaml(geo_parser.run_config_path,
                                         'geo_cslc_s1')

    # Run geocode burst workflow
    run(runconfig)

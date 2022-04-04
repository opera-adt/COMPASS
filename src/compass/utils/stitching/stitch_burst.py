import argparse
import glob
import json
import os
import time

import isce3
import journal
import pandas as pd
import shapely.wkt
from compass.utils import helpers
from osgeo import gdal, ogr
from shapely.geometry import Polygon


def command_line_parser():
    """
    Command line parser
    """

    parser = argparse.ArgumentParser(description="""
                                     Stitch S1-A/B bursts for stack processing""",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--indir', type=str, action='store', dest='indir',
                        help='Directory with S1-A/B bursts organized by dates')
    parser.add_argument('-b', '--burst-list', type=str, nargs='+',
                        default=None, dest='burst_list',
                        help='List of burst IDs to stitch. If None, common bursts'
                             'among all dates will be stitched (default: None')
    parser.add_argument('-m', '--margin', type=float,
                        default=100, dest='margin',
                        help='Margin to apply during stitching. Same units as bursts coordinate system.'
                             '(default: 100 m, UTM)')
    parser.add_argument('-s', '--scratchdir', type=str, default='scratch',
                        dest='scratch',
                        help='Directory where to store temporary results (default: scratch)')
    parser.add_argument('-o', '--outdir', type=str, default='outdir',
                        dest='outdir',
                        help='Directory path where to store stitched bursts (default: outdir)')
    return parser.parse_args()


def main(indir, outdir, scratchdir, margin, burst_list):
    '''
    Stitch S1-A/B bursts for stack processing

    Parameters:
    ----------
    indir: str
       File path to directory containing S1-A/B bursts
       organized by date
    outdir: str
       File path to directory where to store stitched bursts
    scratchdir: str
       File path to directory where to store intermediate
       results (e.g., shapefiles of burst boundary)
    margin: float
       Margin to apply to burst boundary while stitching.
       Same units as bursts coordinate system
    burst_list: list [str]
       List of burst IDs to stitch. If not provided, common
       bursts among all dates are identified and considered
       for stitching
    '''
    info_channel = journal.info('stitch_burst.main')
    error_channel = journal.error('stitch_burst.main')

    # Start tracking time
    info_channel.log('Start burst stitching')
    t_start = time.time()

    # Check that input directory is valid
    if not os.path.isdir(indir):
        err_str = f'{indir} is not a valid input directory'
        error_channel.log(err_str)
        raise ValueError(err_str)

    # Create output and scratch dirs if not existing
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        helpers.check_write_dir(outdir)
    if not os.path.exists(scratchdir):
        os.makedirs(scratchdir, exist_ok=True)
        helpers.check_write_dir(scratchdir)

    # Collect info for stitching in all dirs in 'indir'
    # and return a panda dataframe with info
    data_dict = get_stitching_dict(indir)

    # If stitching some bursts, prune dataframe to
    # contains only the burst IDs to stitch
    if burst_list is not None:
        data_dict = prune_dataframe(data_dict,
                                    'burst_id', burst_list)

    # Identify common burst IDs among different dates
    ids2stitch = get_common_burst_ids(data_dict)

    # Prune dataframe to contain only the IDs to stitch
    data_dict = prune_dataframe(data_dict,
                                'burst_id', ids2stitch)

    # Track cut bursts by adding new column to dataframe
    data_dict["cut_granule_id"] = ""

    # For each burst ID, get common bursts boundary and store it
    # as a shapefile to be used by gdalwarp (later for cutting)
    for burst_id in list(set(data_dict['burst_id'])):
        # Get info on polygons, epsg, granule
        polys = data_dict.polygon[data_dict.burst_id == burst_id].tolist()
        epsgs = data_dict.epsg[data_dict.burst_id == burst_id].tolist()
        granules = data_dict.granule_id[data_dict.burst_id == burst_id].tolist()

        # Get common burst boundary and save it as shapefile
        common_poly, epsg = intersect_polygons(polys, epsgs, margin)
        shp_filename = f'{scratchdir}/shp_{burst_id}.shp'
        save_as_shapefile(common_poly, epsg, shp_filename)

        # Cut all the same ID burts with shapefile
        for granule in granules:
            # Get raster resolution
            xres, yres, epsg = get_raster_info(granule)

            filename = os.path.splitext(os.path.basename(granule))[0]
            cut_filename = f'{scratchdir}/cut_{filename}'
            opts = gdal.WarpOptions(format='ENVI', xRes=xres,
                                    yRes=yres, dstNodata='nan',
                                    cutlineDSName=shp_filename,
                                    multithread=True, warpMemoryLimit=3000,
                                    srcSRS=f'EPSG:{epsg}',
                                    dstSRS=f'EPSG:{epsg}',
                                    targetAlignedPixels=True)
            gdal.Warp(cut_filename, granule, options=opts)
            # Save location of cut burst IDs (later for stitching)
            data_dict.loc[data_dict['granule_id'] == granule,
                          'cut_granule_id'] = cut_filename

    # Start stitching by date
    unique_dates = list(set(data_dict['date']))
    for date in unique_dates:
        cut_rasters = data_dict.cut_granule_id[data_dict.date == date].tolist()
        xres, yres, epsg = get_raster_info(cut_rasters[0])
        outfilename = f'{outdir}/stitched_{date}'
        opts = gdal.WarpOptions(format='ENVI', xRes=xres,
                                yRes=yres, dstNodata='nan',
                                multithread=True, warpMemoryLimit=3000,
                                srcSRS=f'EPSG:{epsg}',
                                dstSRS=f'EPSG:{epsg}',
                                targetAlignedPixels=True)
        gdal.Warp(outfilename, cut_rasters, options=opts)

    # Save data dictionary to keep trace of what has been merged
    data_dict.to_csv(f'{outdir}/merged_report.csv')

    # Log elapsed time for stitching
    dt = time.time() - t_start
    info_channel.log(f'Successfully run stitching in {dt:.3f} seconds')


def get_raster_info(filename):
    '''
    Get raster X and Y resolution and epsg

    Parameters:
    -----------
    filename: str
       Filepath where raster is stored

    Returns:
    -------
    xres, yres, epsg: (tuple, float)
       Raster resolution along X and Y directions and epsg
    '''
    raster = isce3.io.Raster(filename)
    return raster.dx, raster.dy, raster.get_epsg()


def save_as_shapefile(polygon, epsg, outfile):
    '''
    Save polygon as an ESRI shapefile

    Parameters:
    ----------
    polygon: shapely.geometry.Polygon
       Shapely polygon identify burst boundary on the ground
    epsg: int
       EPSG code associate to 'polygon' coordinate system
    outfile: str
       File path to store create ESRI shapefile

    '''
    dest_srs = ogr.osr.SpatialReference()
    dest_srs.ImportFromEPSG(int(epsg))
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.CreateDataSource(outfile)
    layer = ds.CreateLayer('', None, ogr.wkbPolygon)

    # Add attribute and create new feature
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()
    feat = ogr.Feature(defn)
    feat.SetField('id', 123)

    # Make a geometry, from Shapely object
    geom = ogr.CreateGeometryFromWkb(polygon.wkb)
    geom.AssignSpatialReference(dest_srs)
    feat.SetGeometry(geom)
    layer.CreateFeature(feat)

    # Clean up
    feat = geom = None
    ds = layer = feat = geom = None


def intersect_polygons(polygons, epsgs, margin):
    '''
    Get the intersection of polygons stored in 'polygons'

    Parameters:
    -----------
    polygons: list
       List of shapely polygons to intersect
    epsgs: list
       List of EPSGs associate to 'polygons'

    Returns:
    -------
    poly_int: shapely.Polygon
       Result of polygon intersection
    epsg_int: int
       EPSG code associated to poly
    '''
    # Assert validity of inputs
    assert (len(polygons) == len(epsgs))

    # Initialize polygon and epsg of intersection
    poly_int = polygons[0]
    epsg_int = epsgs[0]

    # Initialize coordinate transformation in case
    # there are polygons with different epsgs
    llh = ogr.osr.SpatialReference()
    llh.ImportFromEPSG(epsg_int)
    tgt = ogr.osr.SpatialReference()

    for poly, epsg in zip(polygons, epsgs):
        if epsg != epsg_int:
            # Transform polygons in same coord system as epsg_int
            tgt_x, tgt_y = [], []
            x, y = poly.exterior.coords.xy
            tgt.ImportFromEPSG(epsg)
            trans = ogr.osr.CoordinateTransformation(llh, tgt)
            for lx, ly in zip(x, y):
                dummy_x, dummy_y, dummy_z = trans.Transform(lx, ly, 0)
                tgt_x.append(dummy_x)
                tgt_y.append(dummy_y)
            poly = Polygon(list(zip(tgt_x, tgt_y)))
        poly_int = poly.intersection(poly_int)

    # To be conservative, apply some margin to the polygon (TO DO:
    # check eps)
    poly_int = poly_int.buffer(-margin)
    return poly_int, epsg_int


def get_stitching_dict(indir):
    '''
    Collect info on bursts to stitch and store them
    in a panda dataframe

    Parameters:
    ----------
    indir: str
       Directory where bursts are stored (organized by date)

    Returns:
    -------
    cfg: pandas.DataFrame
       Dataframe collecting info on bursts to stitch
    '''
    # Create dictionary where to store results
    cfg = {'burst_id': [], 'granule_id': [], 'polygon': [],
           'date': [], 'epsg': []}
    # Get list of directory under dir_list
    dir_list = os.listdir(indir)
    for dir in dir_list:
        # List metadata files in the directory
        meta_list = sorted(glob.glob(f'{indir}/{dir}/*json'))
        for path in meta_list:
            # Read metadata file
            metadata = read_metadata(path)
            # Read info and store in dictionary
            cfg['burst_id'].append(get_metadata(metadata, 'burst_id'))
            filename = get_metadata(metadata, 'granule_id')
            cfg['granule_id'].append(f'{indir}/{dir}/{filename}')
            poly = get_metadata(metadata, 'polygon')
            cfg['polygon'].append(shapely.wkt.loads(poly))
            cfg['date'].append(get_metadata(metadata, 'date'))
            cfg['epsg'].append(get_metadata(metadata, 'epsg'))

    return pd.DataFrame(data=cfg)


def read_metadata(meta_file):
    '''Read metadata file in a dictionary

    Parameters:
    -----------
    meta_file: str
       Filepath where metadata is located

    Returns:
    -------
    cfg: dict
       Dictionary containing metadata
    '''
    with open(meta_file) as json_file:
        metadata = json.load(json_file)
    return metadata


def get_metadata(metadata, field):
    '''
    Get 'field" value from metadata dictionary

    Parameters:
    -----------
    metadata: dict
       Dictionary containing metadata for a burst
    field: str
       Field in the metadata to extract value for

    Returns:
    -------
    value: float, int, shapely.Polygon
       Value stored in the metadata field (type
       depends on type of metadata extracted)
    '''
    value = metadata[field]
    return value


def prune_dataframe(data, id_col, id_list):
    '''
    Prune dataframe based on column ID and list of value

    Parameters:
    ----------
    data: pandas.DataFrame
       dataframe that needs to be pruned
    id_col: str
       column identification for 'data' (e.g. 'burst_id')
    id_list: list
       List of elements to keep after pruning. Elements not
       in id_list but contained in 'data' will be pruned

    Returns:
    -------
    data: pandas.DataFrame
       Pruned dataframe with rows in 'id_list'
    '''
    pattern = '|'.join(id_list)
    dataf = data.loc[data[id_col].str.contains(pattern,
                                               case=False)]
    return dataf


def get_common_burst_ids(data):
    '''
    Get list of burst IDs common among all processed dates

    Parameters:
    ----------
    data: pandas.DataFrame
      Dataframe containing info for stitching (e.g. burst IDs)

    Returns:
    -------
    common_id: list
      List containing common burst IDs among all the dates
    '''
    # Identify all the dates for the bursts to stitch
    unique_dates = list(set(data['date']))

    # Initialize list of unique burst IDs
    common_id = data.burst_id[data.date == unique_dates[0]]

    for date in unique_dates:
        ids = data.burst_id[data.date == date]
        common_id = sorted(list(set(ids.tolist()) & set(common_id)))
    return common_id


if __name__ == '__main__':
    # Get command line arguments
    opts = command_line_parser()
    # Give these arguments to the main code
    main(opts.indir, opts.outdir,
         opts.scratch, opts.margin, opts.burst_list)

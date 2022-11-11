'''
Collection of function for determining and setting the geogrid
'''

import numpy as np
import journal

from nisar.workflows.geogrid import _grid_size
import isce3

from compass.utils import helpers

def assign_check_epsg(epsg, epsg_default):
    '''
    Assign and check user-defined epsg

    Parameters
    ----------
    epsg: int
        User-defined EPSG code to check
    epsg_default: int
        Default epsg code to assign

    Returns
    -------
    epsg: int
        Checked EPSG code to use in geogrid
    '''
    if epsg is None: epsg = epsg_default
    assert 1024 <= epsg <= 32767
    return epsg


def assign_check_spacing(x_spacing, y_spacing,
                         x_default_spacing, y_default_spacing):
    '''
    Check validity of input spacings and assign default spacings
    if one or both input spacings are None

    Parameters
    ----------
    x_spacing: float
        Input spacing of the geogrid along X-direction
    y_spacing: float
        Input spacing of the geogrid along Y-direction
    x_default_spacing: float
        Default spacing of the geogrid along X-direction
    y_default_spacing: float
        Default spacing of the geogrid along Y-direction

    Returns
    -------
    x_spacing: float
        Verified geogrid spacing along X-direction
    y_spacing: float
        Verified geogrid spacing along Y-direction
    '''
    error_channel = journal.error('geogrid.assign_check_spacing')

    # Check tha x-y_spacings are valid (positive)
    if y_spacing is not None:
        assert y_spacing > 0.0
        y_spacing = -1.0 * y_spacing
    else:
        assert y_default_spacing > 0.0
        y_spacing = -1.0 * y_default_spacing
    if x_spacing is not None:
        assert x_spacing > 0.0
    else:
        assert x_default_spacing > 0.0
        x_spacing = x_default_spacing

    # Check that x-y_spacings have been correctly assigned
    # (check on default spacings)
    if x_spacing <= 0:
        err_str = f'Pixel spacing in X/longitude direction needs to be >=0 (x_spacing: {x_spacing})'
        error_channel.log(err_str)
        raise ValueError(err_str)
    if y_spacing >= 0:
        err_str = f'Pixel spacing in Y/latitude direction needs to be <=0 (y_spacing: {y_spacing})'
        error_channel.log(err_str)
        raise ValueError(err_str)
    return x_spacing, y_spacing


def assign_check_geogrid(geo_grid, x_start=None, y_start=None,
                         x_end=None, y_end=None):
    '''
    Initialize geogrid with user defined parameters.
    Check the validity of user-defined parameters

    Parameters
    ----------
    geo_grid: isce3.product.geogrid
        ISCE3 object defining the geogrid
    x_start: float
        Geogrid top-left X coordinate
    y_start: float
        Geogrid top-left Y coordinate
    x_end: float
        Geogrid bottom-right X coordinate
    y_end: float
        Geogrid bottom-right Y coordinate

    Returns
    -------
    geo_grid: isce3.product.geogrid
        ISCE3 geogrid initialized with user-defined inputs
    '''

    # Check assigned input coordinates and initialize geogrid accordingly
    if None in [x_start, y_start, x_end, y_end]:
        if x_start is not None:
            new_end_x = geo_grid.start_x + geo_grid.spacing_x * geo_grid.width
            geo_grid.start_x = x_start
            geo_grid.width = int(np.ceil((new_end_x - x_start) /
                                         geo_grid.spacing_x))
        # Restore geogrid end point if provided by the user
        if x_end is not None:
            geo_grid.width = int(np.ceil((x_end - geo_grid.start_x) /
                                         geo_grid.spacing_x))
        if y_start is not None:
            new_end_y = geo_grid.start_y + geo_grid.spacing_y * geo_grid.length
            geo_grid.start_y = y_start
            geo_grid.length = int(np.ceil((new_end_y - y_start) /
                                          geo_grid.spacing_y))
        if y_end is not None:
            geo_grid.length = int(np.ceil((y_end - geo_grid.start_y) /
                                          geo_grid.spacing_y))
    else:
        # If all the start/end coordinates have been assigned,
        # initialize the geogrid with them
        width = _grid_size(x_end, x_start, geo_grid.spacing_x)
        length = _grid_size(y_end, y_start, geo_grid.spacing_y)
        geo_grid = isce3.product.GeoGridParameters(x_start, y_start,
                                                   geo_grid.spacing_x,
                                                   geo_grid.spacing_y,
                                                   width, length,
                                                   geo_grid.epsg)
    return geo_grid


def check_geogrid_endpoints(geo_grid, x_end=None, y_end=None):
    '''
    Check validity of geogrid end points

    Parameters
    -----------
    geo_grid: isce3.product.geogrid
        ISCE3 object defining the geogrid
    x_end: float
        Geogrid bottom right X coordinate
    y_end: float
        Geogrid bottom right Y coordinates

    Returns
    -------
    x_end: float
        Verified geogrid bottom-right X coordinate
    y_end: float
        Verified geogrid bottom-right Y coordinate
    '''
    end_pt = lambda start, sz, spacing: start + spacing * sz

    if x_end is None:
        x_end = end_pt(geo_grid.start_x, geo_grid.spacing_x, geo_grid.width)
    if y_end is None:
        y_end = end_pt(geo_grid.start_y, geo_grid.spacing_y, geo_grid.length)
    return x_end, y_end


def check_snap_values(x_snap, y_snap, x_spacing, y_spacing):
    '''
    Check validity of snap values

    Parameters
    ----------
    x_snap: float
        Snap value along X-direction
    y_snap: float
        Snap value along Y-direction
    x_spacing: float
        Spacing of the geogrid along X-direction
    y_spacing: float
        Spacing of the geogrid along Y-direction
    '''
    error_channel = journal.error('geogrid.check_snap_values')

    # Check that snap values in X/Y-directions are positive
    if x_snap is not None and x_snap <= 0:
        err_str = f'Snap value in X direction must be > 0 (x_snap: {x_snap})'
        error_channel.log(err_str)
        raise ValueError(err_str)
    if y_snap is not None and y_snap <= 0:
        err_str = f'Snap value in Y direction must be > 0 (y_snap: {y_snap})'
        error_channel.log(err_str)
        raise ValueError(err_str)

    # Check that snap values in X/Y are integer multiples of the geogrid
    # spacings in X/Y directions
    if x_snap is not None and x_snap % x_spacing != 0.0:
        err_str = 'x_snap must be exact multiple of spacing in X direction (x_snap % x_spacing !=0)'
        error_channel.log(err_str)
        raise ValueError(err_str)
    if y_snap is not None and y_snap % y_spacing != 0.0:
        err_str = 'y_snap must be exact multiple of spacing in Y direction (y_snap % y_spacing !=0)'
        error_channel.log(err_str)
        raise ValueError(err_str)


def snap_geogrid(geo_grid, x_snap, y_snap, x_end, y_end):
    '''
    Snap geogrid based on user-defined snapping values

    Parameters
    ----------
    geo_grid: isce3.product.geogrid
        ISCE3 object definining the geogrid
    x_snap: float
        Snap value along X-direction
    y_snap: float
        Snap value along Y-direction
    x_end: float
        Bottom-right X coordinate
    y_end: float
        Bottom-right Y coordinate

    Returns
    -------
    geo_grid: isce3.product.geogrid
        ISCE3 object containing the snapped geogrid
    '''
    if x_end is None: x_end = geo_grid.end_x
    if y_end is None: y_end = geo_grid.end_y

    if x_snap is not None or y_snap is not None:
        snap_coord = lambda val, snap, round_func: round_func(
            float(val) / snap) * snap
        geo_grid.start_x = snap_coord(geo_grid.start_x, x_snap, np.floor)
        geo_grid.start_y = snap_coord(geo_grid.start_y, y_snap, np.ceil)
        end_x = snap_coord(x_end, x_snap, np.ceil)
        end_y = snap_coord(y_end, y_snap, np.floor)
        geo_grid.length = _grid_size(end_y, geo_grid.start_y,
                                     geo_grid.spacing_y)
        geo_grid.width = _grid_size(end_x, geo_grid.start_x, geo_grid.spacing_x)
    return geo_grid


def get_point_epsg(lat, lon):
    '''
    Get EPSG code based on latitude and longitude
    coordinates of a point

    Parameters
    ----------
    lat: float
        Latitude coordinate of the point
    lon: float
        Longitude coordinate of the point

    Returns
    -------
    epsg: int
        UTM zone
    '''
    error_channel = journal.error('geogrid.get_point_epsg')

    if lon >= 180.0:
        lon = lon - 360.0

    if lat >= 60.0:
        return 3413
    elif lat <= -60.0:
        return 3031
    elif lat > 0:
        return 32601 + int(np.round((lon + 177) / 6.0))
    elif lat < 0:
        return 32701 + int(np.round((lon + 177) / 6.0))
    else:
        err_str = "'Could not determine EPSG for {0}, {1}'.format(lon, lat))"
        error_channel.log(err_str)
        raise ValueError(err_str)


def generate_geogrids_from_db(bursts, geo_dict, dem, burst_db_file):
    ''' Create a geogrid for all bursts in given list

    Parameters
    ----------
    burst: list[Sentinel1BurstSlc]
        List of bursts
    geo_dict: dict
        Dict of parameters that describe the area to be geocoded
    dem: str
        Path to DEM raster
    burst_db_file : str
        Location of burst database sqlite file

    Returns
    -------
    geo_grids: dict
        Dict of burst ID keys to isce3.product.GeoGridParameters values
    '''
    dem_raster = isce3.io.Raster(dem)

    # Unpack values from geocoding dictionary
    x_spacing_dict = geo_dict['x_posting']
    y_spacing_dict = geo_dict['y_posting']
    x_snap_dict = geo_dict['x_snap']
    y_snap_dict = geo_dict['y_snap']

    geo_grids = {}

    # get all burst IDs and their EPSGs + bounding boxes
    burst_ids = set([b.burst_id for b in bursts])
    epsgs, bboxes, _ = helpers.burst_bbox_from_db(burst_ids, burst_db_file)
    epsg_bbox_dict = dict(zip(burst_ids, zip(epsgs, bboxes)))

    for burst in bursts:
        burst_id = burst.burst_id

        # check geogrid already created for burst ID
        if burst_id in geo_grids:
            continue

        # extract EPSG and bbox for current burst from dict
        # bottom right = (xmax, ymin) and top left = (xmin, ymax)
        epsg, (xmin, ymin, xmax, ymax) = epsg_bbox_dict[burst_id]

        radar_grid = burst.as_isce3_radargrid()
        orbit = burst.orbit

        # Check spacing in X/Y direction
        if epsg == dem_raster.get_epsg():
            x_spacing, y_spacing = assign_check_spacing(x_spacing_dict,
                                                        y_spacing_dict,
                                                        4.5e-5, 9.0e-5)
        else:
            # Assign spacing in meters
            x_spacing, y_spacing = assign_check_spacing(x_spacing_dict,
                                                        y_spacing_dict,
                                                        5.0, 10.0)

        # Initialize geogrid with the info checked at this stage
        geo_grid_in = isce3.product.bbox_to_geogrid(radar_grid, orbit,
                                                    isce3.core.LUT2d(),
                                                    x_spacing, y_spacing, epsg)
        # Check and further initialize geo_grid
        geo_grid = assign_check_geogrid(geo_grid_in, xmin, ymax, xmax, ymin)

        # Check end point of geogrid before compute snaps
        x_end, y_end = check_geogrid_endpoints(geo_grid, xmax, ymin)
        # Check snap values
        check_snap_values(x_snap_dict, y_snap_dict, x_spacing, y_spacing)
        # Snap coordinates
        geo_grid = snap_geogrid(geo_grid, x_snap_dict, y_snap_dict, x_end, y_end)

        geo_grids[burst_id] = geo_grid

    return geo_grids


def generate_geogrids(bursts, geo_dict, dem):
    ''' Create a geogrid for all bursts in given list

    Parameters
    ----------
    burst: list[Sentinel1BurstSlc]
        List of bursts
    geo_dict: dict
        Dict of parameters that describe the area to be geocoded
    dem: str
        Path to DEM raster

    Returns
    -------
    geo_grids: dict
        Dict of burst ID keys to isce3.product.GeoGridParameters values
    '''
    dem_raster = isce3.io.Raster(dem)

    # Unpack values from geocoding dictionary
    epsg_dict = geo_dict['output_epsg']
    x_start_dict = geo_dict['top_left']['x']
    y_start_dict = geo_dict['top_left']['y']
    x_spacing_dict = geo_dict['x_posting']
    y_spacing_dict = geo_dict['y_posting']
    x_end_dict = geo_dict['bottom_right']['x']
    y_end_dict = geo_dict['bottom_right']['y']
    x_snap_dict = geo_dict['x_snap']
    y_snap_dict = geo_dict['y_snap']

    geo_grids = {}
    for burst in bursts:
        burst_id = burst.burst_id

        if burst_id in geo_grids:
            continue

        # Compute Burst epsg if not assigned in runconfig
        epsg_default = get_point_epsg(burst.center.y,
                                      burst.center.x)
        epsg = assign_check_epsg(epsg_dict, epsg_default)

        radar_grid = burst.as_isce3_radargrid()
        orbit = burst.orbit

        # Check spacing in X/Y direction
        if epsg == dem_raster.get_epsg():
            x_spacing, y_spacing = assign_check_spacing(x_spacing_dict,
                                                        y_spacing_dict,
                                                        4.5e-5, 9.0e-5)
        else:
            # Assign spacing in meters
            x_spacing, y_spacing = assign_check_spacing(x_spacing_dict,
                                                        y_spacing_dict,
                                                        5.0, 10.0)

        # Initialize geogrid with the info checked at this stage
        geo_grid_in = isce3.product.bbox_to_geogrid(radar_grid, orbit,
                                                    isce3.core.LUT2d(),
                                                    x_spacing, y_spacing, epsg)
        # Check and further initialize geo_grid
        geo_grid = assign_check_geogrid(geo_grid_in, x_start_dict,
                                        y_start_dict, x_end_dict,
                                        y_end_dict)

        # Check end point of geogrid before compute snaps
        x_end, y_end = check_geogrid_endpoints(geo_grid, x_end_dict, y_end_dict)
        # Check snap values
        check_snap_values(x_snap_dict, y_snap_dict, x_spacing, y_spacing)
        # Snap coordinates
        geo_grid = snap_geogrid(geo_grid, x_snap_dict, y_snap_dict, x_end, y_end)

        geo_grids[burst_id] = geo_grid

    return geo_grids


def geogrid_as_dict(grid):
    geogrid_dict = {attr:getattr(grid, attr) for attr in grid.__dir__()
                    if attr != 'print' and attr[:2] != '__'}
    return geogrid_dict

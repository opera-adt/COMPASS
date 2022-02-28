'''
Collection of function for determining and setting the geogrid
'''

import numpy as np
import journal
from osgeo import osr

from nisar.workflows.geogrid import _grid_size
import isce3


def assign_check_epsg(epsg, epsg_default):
    if epsg is None: epsg = epsg_default
    assert 1024 <= epsg <= 32767
    return epsg


def assign_check_spacing(x_spacing, y_spacing,
                         x_default_spacing, y_default_spacing):
    error_channel = journal.error('geogrid.assign_check_spacing')
    # Check if spacing are correctly assigned
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
    if None in [x_start, y_start, x_end, y_end]:
        if x_start is not None:
            new_end_x = geo_grid.start_x + geo_grid.spacing_x * geo_grid.width
            geo_grid.start_x = x_start
            geo_grid.width = int(np.ceil((new_end_x - x_start) /
                                         geo_grid.spacing_x))
            # Restore user-defined x_end if provided
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
        # All info have been assigned. Modify geogrid accordingly
        width = _grid_size(x_end, x_start, geo_grid.spacing_x)
        length = _grid_size(y_end, y_start, geo_grid.spacing_y)
        geo_grid = isce3.product.GeoGridParameters(x_start, y_start,
                                                   geo_grid.spacing_x,
                                                   geo_grid.spacing_y,
                                                   width, length,
                                                   geo_grid.epsg)

    return geo_grid


def check_geogrid_endpoints(geo_grid, x_end=None, y_end=None):
    end_pt = lambda start, sz, spacing: start + spacing * sz

    if x_end is None:
        x_end = end_pt(geo_grid.start_x, geo_grid.spacing_x, geo_grid.width)
    if y_end is None:
        y_end = end_pt(geo_grid.start_y, geo_grid.spacing_y, geo_grid.length)
    return x_end, y_end


def check_snap_values(x_snap, y_snap, x_spacing, y_spacing):
    error_channel = journal.error('geogrid.check_snap_values')

    # Check that snap values are >=0
    if x_snap is not None and x_snap <= 0:
        err_str = f'Snap value in X direction must be > 0 (x_snap: {x_snap})'
        error_channel.log(err_str)
        raise ValueError(err_str)

    if y_snap is not None and y_snap <= 0:
        err_str = f'Snap value in Y direction must be > 0 (y_snap: {y_snap})'
        error_channel.log(err_str)
        raise ValueError(err_str)

    # Check that snap values are integer multiples of spacings
    if x_snap is not None and x_snap % x_spacing != 0.0:
        err_str = f'x_snap must be exact multiple of spacing in X direction (x_snap % x_spacing !=0)'
        error_channel.log(err_str)
        raise ValueError(err_str)
    if y_snap is not None and y_snap % y_spacing != 0.0:
        err_str = f'y_snap must be exact multiple of spacing in Y direction (y_snap % y_spacing !=0)'
        error_channel.log(err_str)
        raise ValueError(err_str)


def snap_geogrid(geo_grid, x_snap, y_snap, x_end, y_end):
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


def generate_geogrid(radar_grid, orbit, dem_raster, geo_dict):
    error_channel = journal.error('geogrid.create_geogrid')

    # Unpack values from geocoding disctionary
    epsg_dict = geo_dict.output_epsg
    x_start_dict = geo_dict.top_left.x
    y_start_dict = geo_dict.top_left.y
    x_spacing_dict = geo_dict.x_posting
    y_spacing_dict = geo_dict.y_posting
    x_end_dict = geo_dict.bottom_right.x
    y_end_dict = geo_dict.bottom_right.y
    x_snap_dict = geo_dict.x_snap
    y_snap_dict = geo_dict.y_snap

    # Check epsg. If None, assign DEM epsg
    epsg = assign_check_epsg(epsg_dict, dem_raster.get_epsg())

    # Check spacing in X/Y direction
    if epsg == dem_raster.get_epsg():
        x_spacing, y_spacing = assign_check_spacing(x_spacing_dict,
                                                    y_spacing_dict,
                                                    dem_raster.dx,
                                                    dem_raster.dy)
    else:
        # Assign default spacing based on the selected epsg
        epsg_spatial_ref = osr.SpatialReference()
        epsg_spatial_ref.ImportFromEPSG(epsg)
        if epsg_spatial_ref.IsGeographic():
            # Assign lat/lon default spacings in degrees
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
    return geo_grid

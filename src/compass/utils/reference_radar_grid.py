import isce3


def file_to_rdr_grid(ref_grid_path: str) -> isce3.product.RadarGridParameters:
    '''read parameters from text file needed to create radar grid object'''
    with open(ref_grid_path, 'r') as f_rdr_grid:
        sensing_start = float(f_rdr_grid.readline())
        wavelength = float(f_rdr_grid.readline())
        prf = float(f_rdr_grid.readline())
        starting_range = float(f_rdr_grid.readline())
        range_pixel_spacing = float(f_rdr_grid.readline())
        length = int(f_rdr_grid.readline())
        width = int(f_rdr_grid.readline())
        # read date string and remove newline
        date_str = f_rdr_grid.readline()
        ref_epoch = isce3.core.DateTime(date_str[:-1])

        rdr_grid = isce3.product.RadarGridParameters(
            sensing_start, wavelength, prf, starting_range,
            range_pixel_spacing, "right", length, width,
            ref_epoch)

        return rdr_grid


def rdr_grid_to_file(ref_grid_path: str,
                     rdr_grid: isce3.product.RadarGridParameters) -> None:
    '''save parameters needed to create a new radar grid object'''
    with open(ref_grid_path, "w") as f_rdr_grid:
        f_rdr_grid.write(str(rdr_grid.sensing_start) + '\n')
        f_rdr_grid.write(str(rdr_grid.wavelength) + '\n')
        f_rdr_grid.write(str(rdr_grid.prf) + '\n')
        f_rdr_grid.write(str(rdr_grid.starting_range) + '\n')
        f_rdr_grid.write(str(rdr_grid.range_pixel_spacing) + '\n')
        f_rdr_grid.write(str(rdr_grid.length) + '\n')
        f_rdr_grid.write(str(rdr_grid.width) + '\n')
        f_rdr_grid.write(str(rdr_grid.ref_epoch) + '\n')

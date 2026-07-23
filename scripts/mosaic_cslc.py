#!/usr/bin/env python3
"""
Merge all CSLC-S1 burst products in a directory into one mosaic + browse PNGs.

Each geocoded CSLC burst lives in its own ``.h5`` (``data/<pol>``) on a UTM grid.
GDAL does not read the georeferencing from the product, so this rebuilds each
burst's SRS + GeoTransform from ``data/x_coordinates`` / ``data/y_coordinates`` /
``data/projection``, and paints every burst's valid pixels (``|z| > 0``) into a
shared grid. (``gdal.BuildVRT`` -- the pattern used for the real-valued DEM
tiles in ``compass_batch.staging.dem.build_vrt`` -- cannot be reused here: CSLC
bursts are *complex* and overlap heavily with zero-fill corners, and GDAL's VRT
nodata compositing does not skip those zeros for complex data, so neighbours get
overwritten and the mosaic comes out gridded/gappy.) It then renders:

- ``<name>_mosaic.tif``     -- merged complex CSLC (CFloat32), georeferenced;
- ``<name>_amplitude.png``  -- log-amplitude, percentile-clipped greyscale
                               (the scaling used by ``compass.utils.browse_image``);
- ``<name>_phase.png``      -- wrapped phase on a cyclic colormap.

Bursts must share one EPSG (true for a single frame/track); mixed EPSGs are
reported and would need ``gdal.Warp`` to a common grid.

Examples
--------
    python mosaic_cslc.py output_s1c_post
    python mosaic_cslc.py output_s1c_post -o browse --pol VV --max-dim 3000

"""

from __future__ import annotations

import argparse
import glob
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()


def burst_georef(h5: str, pol: str) -> tuple[int, list[float], int, int]:
    """Return ``(epsg, [ulx, uly, lrx, lry], nx, ny)`` for a CSLC burst."""
    with h5py.File(h5, "r", locking=False) as f:
        epsg = int(f["data/projection"][()])
        x = f["data/x_coordinates"][:]
        y = f["data/y_coordinates"][:]
        ny, nx = f[f"data/{pol}"].shape  # pylint: disable=no-member
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])  # negative (north-up)
    ulx, uly = float(x[0]) - dx / 2, float(y[0]) - dy / 2
    lrx, lry = float(x[-1]) + dx / 2, float(y[-1]) + dy / 2
    return epsg, [ulx, uly, lrx, lry], nx, ny


def build_mosaic(h5s: list[str], pol: str, out_tif: str) -> gdal.Dataset:  # pylint: disable=too-many-locals
    """
    Paint-over mosaic of the complex bursts into a georeferenced GeoTIFF.

    Each geocoded burst is a slanted parallelogram of valid data inside an
    axis-aligned UTM box, so ~60% of the box is zero-fill, and adjacent bursts
    overlap heavily. GDAL's VRT/Warp nodata compositing is unreliable for
    *complex* data: in the overlaps a later burst's zero-fill corner overwrites
    a neighbour's valid pixels, leaving gaps (a gridded, unstitched mosaic).

    Instead we paint each burst's valid pixels (``|z| > 0``) into a shared grid,
    so an overlap keeps whichever burst actually has data. Bursts must share one
    EPSG and pixel spacing (true for a single frame/track); they are assumed to
    lie on a common pixel lattice and are snapped to it by rounding.
    """
    metas, epsgs = [], set()  # meta: (h5, ulx, uly, nx, ny)
    dx = dy = None
    for h5 in h5s:
        try:
            epsg, (ulx, uly, _lrx, _lry), nx, ny = burst_georef(h5, pol)
            with h5py.File(h5, "r", locking=False) as f:
                x = f["data/x_coordinates"]
                y = f["data/y_coordinates"]
                dx = float(x[1] - x[0])
                dy = float(y[1] - y[0])  # negative (north-up)
        except (KeyError, RuntimeError, OSError) as exc:
            print(f"  skip {Path(h5).name}: {exc}")
            continue
        metas.append((h5, ulx, uly, nx, ny))
        epsgs.add(epsg)
    if not metas:
        sys.exit("No readable bursts found.")
    if len(epsgs) > 1:
        sys.exit(f"Bursts span multiple EPSGs {epsgs}; reproject with gdal.Warp first.")
    epsg = epsgs.pop()

    # Global grid: union of all burst boxes on the shared lattice.
    ulx = min(m[1] for m in metas)
    uly = max(m[2] for m in metas)
    lrx = max(m[1] + m[3] * dx for m in metas)
    lry = min(m[2] + m[4] * dy for m in metas)
    w = int(round((lrx - ulx) / dx))
    h = int(round((uly - lry) / -dy))

    ds = gdal.GetDriverByName("GTiff").Create(
        out_tif, w, h, 1, gdal.GDT_CFloat32,
        options=["COMPRESS=DEFLATE", "BIGTIFF=YES", "TILED=YES"],
    )
    ds.SetGeoTransform((ulx, dx, 0.0, uly, 0.0, dy))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(0)

    for h5, bx, by, nx, ny in metas:
        c0 = int(round((bx - ulx) / dx))
        r0 = int(round((uly - by) / -dy))
        with h5py.File(h5, "r", locking=False) as f:
            a = f[f"data/{pol}"][:]  # pylint: disable=no-member
        r1, c1 = min(r0 + ny, h), min(c0 + nx, w)
        a = a[: r1 - r0, : c1 - c0]
        cur = band.ReadAsArray(c0, r0, c1 - c0, r1 - r0)
        valid = np.abs(a) > 0
        cur[valid] = a[valid]  # paint valid pixels only; overlaps keep real data
        band.WriteArray(cur, c0, r0)
    band.FlushCache()
    ds.FlushCache()
    print(f"  mosaicked {len(metas)} burst(s), EPSG {epsg}  grid {w}x{h}")
    return ds


def read_decimated(ds: gdal.Dataset, max_dim: int) -> np.ndarray:
    """Read the (complex) mosaic decimated so the long side <= ``max_dim``."""
    w, h = ds.RasterXSize, ds.RasterYSize
    scale = max(1, int(np.ceil(max(w, h) / max_dim)))
    bw, bh = w // scale, h // scale
    arr = ds.GetRasterBand(1).ReadAsArray(buf_xsize=bw, buf_ysize=bh)
    return arr


def _clip_percentile(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip to the [lo, hi] percentiles of finite pixels (cf. browse_image)."""
    finite = np.isfinite(a)
    vmin, vmax = np.nanpercentile(a[finite], [lo, hi])
    return np.clip(a, vmin, vmax), vmin, vmax


def save_amplitude_png(z: np.ndarray, path: str, lo: float, hi: float) -> None:
    """Render log-amplitude as a percentile-clipped greyscale PNG."""
    import matplotlib.pyplot as plt

    amp = np.abs(z)
    mask = np.isfinite(amp) & (amp > 0)
    log = np.full(amp.shape, np.nan, dtype="float32")
    log[mask] = 20.0 * np.log10(amp[mask])  # dB
    clipped, vmin, vmax = _clip_percentile(log, lo, hi)
    plt.imsave(path, np.ma.masked_invalid(clipped), cmap="gray", vmin=vmin, vmax=vmax)
    print(f"  amplitude PNG -> {path}  ({vmin:.1f}..{vmax:.1f} dB)")


def save_phase_png(z: np.ndarray, path: str) -> None:
    """Render wrapped phase [-pi, pi] on a cyclic colormap."""
    import matplotlib.pyplot as plt

    phase = np.angle(z)
    phase[~(np.isfinite(z) & (np.abs(z) > 0))] = np.nan
    plt.imsave(path, np.ma.masked_invalid(phase), cmap="twilight_shifted",
               vmin=-np.pi, vmax=np.pi)
    print(f"  phase PNG     -> {path}")


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and mosaic the CSLC bursts into a single raster."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cslc_dir", help="Directory of CSLC burst products (searched for *.h5).")
    p.add_argument("-o", "--out-dir", default=None,
                   help="Output directory (default: <cslc_dir>).")
    p.add_argument("--pol", default="VV", help="Polarization dataset (default: VV).")
    p.add_argument("--max-dim", type=int, default=2048,
                   help="Longest PNG dimension in pixels (default: 2048).")
    p.add_argument("--pct", type=float, nargs=2, metavar=("LO", "HI"),
                   default=(2.0, 98.0), help="Amplitude clip percentiles.")
    p.add_argument("--no-tif", action="store_true",
                   help="Skip writing the merged complex GeoTIFF.")
    args = p.parse_args(argv)

    cslc_dir = Path(args.cslc_dir)
    out_dir = Path(args.out_dir) if args.out_dir else cslc_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cslc_dir.name.replace("output_", "") or cslc_dir.name

    h5s = sorted(glob.glob(str(cslc_dir / "**" / "*.h5"), recursive=True))
    if not h5s:
        sys.exit(f"No .h5 products under {cslc_dir}")
    print(f"Merging {len(h5s)} burst(s) from {cslc_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        # build_mosaic writes a GeoTIFF; keep it as the product unless --no-tif,
        # in which case it goes to a scratch file we read for the browse only.
        tif = str(Path(tmp) / "mosaic.tif") if args.no_tif \
            else str(out_dir / f"{name}_mosaic.tif")
        mosaic = build_mosaic(h5s, args.pol, tif)
        if not args.no_tif:
            print(f"  merged complex GeoTIFF -> {tif}")
        z = read_decimated(mosaic, args.max_dim)
        mosaic = None  # close dataset before the scratch dir is removed

    save_amplitude_png(z, str(out_dir / f"{name}_amplitude.png"), *args.pct)
    save_phase_png(z, str(out_dir / f"{name}_phase.png"))
    print("Done.")


if __name__ == "__main__":
    main()

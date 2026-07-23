#!/usr/bin/env python3
"""Merge all CSLC-S1 burst products in a directory into one mosaic + browse PNGs.

Each geocoded CSLC burst lives in its own ``.h5`` (``data/<pol>``) on a UTM grid.
GDAL does not read the georeferencing from the product, so this rebuilds each
burst's SRS + GeoTransform from ``data/x_coordinates`` / ``data/y_coordinates`` /
``data/projection``, mosaics the bursts with ``gdal.BuildVRT`` (the same VRT
mosaic pattern as ``compass_batch.staging.dem.build_vrt``), and renders:

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
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def burst_georef(h5: str, pol: str) -> tuple[int, list[float], int, int]:
    """Return ``(epsg, [ulx, uly, lrx, lry], nx, ny)`` for a CSLC burst."""
    with h5py.File(h5, "r", locking=False) as f:
        epsg = int(f["data/projection"][()])
        x = f[f"data/x_coordinates"][:]
        y = f["data/y_coordinates"][:]
        ny, nx = f[f"data/{pol}"].shape
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])  # negative (north-up)
    ulx, uly = float(x[0]) - dx / 2, float(y[0]) - dy / 2
    lrx, lry = float(x[-1]) + dx / 2, float(y[-1]) + dy / 2
    return epsg, [ulx, uly, lrx, lry], nx, ny


def burst_vrt(h5: str, pol: str, vrt_dir: Path) -> tuple[str, int]:
    """Write a georeferenced VRT wrapping the burst's complex ``data/<pol>``."""
    epsg, bounds, _, _ = burst_georef(h5, pol)
    src = f'HDF5:"{h5}"://data/{pol}'
    vrt = str(vrt_dir / (Path(h5).stem + f"_{pol}.vrt"))
    gdal.Translate(
        vrt,
        gdal.Open(src),
        format="VRT",
        outputSRS=f"EPSG:{epsg}",
        outputBounds=bounds,  # [ulx, uly, lrx, lry]
        noData=0,
    )
    return vrt, epsg


def build_mosaic(h5s: list[str], pol: str, vrt_dir: Path) -> gdal.Dataset:
    """Mosaic the burst products into a single in-memory complex VRT."""
    vrts, epsgs = [], set()
    for h5 in h5s:
        try:
            vrt, epsg = burst_vrt(h5, pol, vrt_dir)
        except (KeyError, RuntimeError) as exc:
            print(f"  skip {Path(h5).name}: {exc}")
            continue
        vrts.append(vrt)
        epsgs.add(epsg)
    if not vrts:
        sys.exit("No readable bursts found.")
    if len(epsgs) > 1:
        sys.exit(f"Bursts span multiple EPSGs {epsgs}; reproject with gdal.Warp first.")
    mosaic = str(vrt_dir / "mosaic.vrt")
    gdal.BuildVRT(mosaic, vrts, srcNodata=0, VRTNodata=0)
    print(f"  mosaicked {len(vrts)} burst(s), EPSG {epsgs.pop()}")
    return gdal.Open(mosaic)


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

    h5s = sorted(glob.glob(str(cslc_dir / "**" / f"*.h5"), recursive=True))
    if not h5s:
        sys.exit(f"No .h5 products under {cslc_dir}")
    print(f"Merging {len(h5s)} burst(s) from {cslc_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        mosaic = build_mosaic(h5s, args.pol, Path(tmp))
        if not args.no_tif:
            tif = str(out_dir / f"{name}_mosaic.tif")
            gdal.Translate(tif, mosaic, format="GTiff",
                           creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"])
            print(f"  merged complex GeoTIFF -> {tif}")
        z = read_decimated(mosaic, args.max_dim)

    save_amplitude_png(z, str(out_dir / f"{name}_amplitude.png"), *args.pct)
    save_phase_png(z, str(out_dir / f"{name}_phase.png"))
    print("Done.")


if __name__ == "__main__":
    main()

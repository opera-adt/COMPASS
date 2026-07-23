#!/usr/bin/env python3
"""Stage every input the CSLC-S1 SAS runconfig needs, into a local ``input_data`` dir.

Given one Sentinel-1 SLC granule name this fetches the five inputs referenced by
``runconfig_cslc_s1_*.yaml`` and writes matching runconfig(s):

1. SLC SAFE zip          -- ASF DAAC HTTPS datapool (Earthdata Login).
2. Precise orbit (EOF)   -- ASF S1 aux archive, matched by validity window
                            (POEORB, falling back to RESORB for recent dates).
3. DEM (ellipsoidal)     -- Copernicus GLO-30 stitched for the granule footprint
                            with ``dem_stitcher`` (geoid -> ellipsoid corrected).
4. Ionosphere (IONEX TEC)-- NASA CDDIS GNSS IONEX; defaults to the Rapid IGS
                            (IGR) solution the CSLC-S1-SAS uses.
5. Burst database        -- OPERA bbox-only SQLite (public burst_db release).

The script is deliberately standalone: it imports only the packages already in
the ``compass`` environment (``requests``, ``dem_stitcher``, ``rasterio`` /
``osgeo``, ``ruamel.yaml``, ``shapely``) so it adds no new dependency and needs
neither ``compass``, ``s1reader``, ``sentineleof`` nor ``asf_search``.

Credentials
-----------
ASF and CDDIS use NASA Earthdata Login. Provide it once in ``~/.netrc``::

    machine urs.earthdata.nasa.gov login <user> password <pass>

Examples
--------
Stage everything for one granule and write both runconfigs::

    python stage_cslc_inputs.py all \
        S1A_IW_SLC__1SDV_20220501T015035_20220501T015102_043011_0522A4_42CC

Stage only the ionosphere file with the final (rather than rapid) solution::

    python stage_cslc_inputs.py iono <GRANULE> --product-type FINAL
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import netrc
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
EDL_HOST = "urs.earthdata.nasa.gov"
ASF_DATAPOOL = "https://datapool.asf.alaska.edu/SLC"
ASF_SEARCH_API = "https://api.daac.asf.alaska.edu/services/search/param"
ASF_AUX = "https://s1qc.asf.alaska.edu"
CDDIS_IONEX = "https://cddis.nasa.gov/archive/gnss/products/ionex"

# Public OPERA burst database (bbox-only SQLite the SAS reads for burst grids).
DEFAULT_BURST_DB_URL = (
    "https://github.com/opera-adt/burst_db/releases/download/"
    "v0.10.0/opera-burst-bbox-only.sqlite3"
)

# CDDIS switched to the long IGS product names on this date.
_NEW_IONEX_NAME_FROM = datetime.date(2023, 10, 18)

_CHUNK = 8 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Earthdata-authenticated HTTP session
# --------------------------------------------------------------------------- #
class _EarthdataSession(requests.Session):
    """Session that (re)applies EDL Basic-Auth on every hop to Earthdata.

    ``requests`` does not carry the Authorization header across redirects, and
    the ASF datapool bounces through an intermediate host
    (``datapool.asf`` -> ``sentinel1.asf/login`` -> ``urs.earthdata``) that
    strips it before Earthdata is reached, yielding a 401 at the OAuth
    ``authorize`` step. So rather than only *preserving* the header, re-apply it
    whenever the redirect target is ``urs.earthdata.nasa.gov`` (and strip it on
    any other host, to avoid leaking the credentials to the data hosts).
    """

    def __init__(self, username: str, password: str):
        super().__init__()
        self._creds = (username, password)
        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response) -> None:  # noqa: D102
        host = urlparse(prepared_request.url).hostname
        if host == EDL_HOST:
            prepared_request.prepare_auth(self._creds)
        elif "Authorization" in prepared_request.headers:
            del prepared_request.headers["Authorization"]


def earthdata_session() -> _EarthdataSession:
    """Build an Earthdata session from ``$EARTHDATA_*`` env vars or ``~/.netrc``."""
    user = os.environ.get("EARTHDATA_USERNAME")
    passwd = os.environ.get("EARTHDATA_PASSWORD")
    if not (user and passwd):
        auth = netrc.netrc().authenticators(EDL_HOST)
        if not auth:
            sys.exit(
                f"No Earthdata credentials: set $EARTHDATA_USERNAME/"
                f"$EARTHDATA_PASSWORD or add 'machine {EDL_HOST}' to ~/.netrc"
            )
        user, _, passwd = auth
    return _EarthdataSession(user, passwd)


def _stream_to_file(session: requests.Session, url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` (streamed), raising on HTTP error."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with session.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                fh.write(chunk)
    tmp.rename(dest)


# --------------------------------------------------------------------------- #
# Granule-name parsing
# --------------------------------------------------------------------------- #
def _clean_name(granule: str) -> str:
    """Strip path, ``.zip`` / ``.SAFE`` suffix from a granule name."""
    stem = Path(granule).name
    for suffix in (".zip", ".SAFE"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def parse_granule(granule: str) -> tuple[str, datetime.datetime, datetime.datetime]:
    """Return ``(mission, sensing_start, sensing_stop)`` for a S1 granule name."""
    name = _clean_name(granule)
    parts = name.split("_")
    if len(parts) < 7 or not parts[0].startswith("S1"):
        sys.exit(f"Not a Sentinel-1 SLC granule name: {granule!r}")
    fmt = "%Y%m%dT%H%M%S"
    start = datetime.datetime.strptime(parts[5], fmt)
    stop = datetime.datetime.strptime(parts[6], fmt)
    return parts[0], start, stop


# --------------------------------------------------------------------------- #
# 1. SLC SAFE zip
# --------------------------------------------------------------------------- #
def stage_slc(granule: str, out_dir: Path, overwrite: bool = False) -> Path:
    """Download the full SAFE zip from ASF's HTTPS datapool."""
    name = _clean_name(granule)
    mission = name[2]  # 'A' / 'C' / 'D'
    url = f"{ASF_DATAPOOL}/S{mission}/{name}.zip"
    dest = out_dir / f"{name}.zip"
    if dest.exists() and not overwrite:
        print(f"  SLC present, skip: {dest.name}")
        return dest
    print(f"  SLC download {url}")
    _stream_to_file(earthdata_session(), url, dest)
    print(f"  -> {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
    return dest


# --------------------------------------------------------------------------- #
# 2. Orbit (EOF)
# --------------------------------------------------------------------------- #
_EOF_RE = re.compile(
    r"(?P<mission>S1[ABCD])_OPER_AUX_(?P<kind>POEORB|RESORB)_OPOD_\d{8}T\d{6}"
    r"_V(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})\.EOF"
)


def _list_orbits(session: requests.Session, kind: str) -> list[str]:
    """Return every orbit ``.EOF`` filename in the ASF aux archive for ``kind``."""
    sub = "aux_poeorb" if kind == "POEORB" else "aux_resorb"
    resp = session.get(f"{ASF_AUX}/{sub}/", timeout=120)
    resp.raise_for_status()
    return sorted(set(re.findall(rf"S1[ABCD]_OPER_AUX_{kind}_OPOD_[0-9T_V]+\.EOF", resp.text)))


def _match_orbit(
    names: list[str], mission: str, start: datetime.datetime, stop: datetime.datetime
) -> str | None:
    """Return the best orbit filename whose validity window covers ``[start, stop]``.

    A single POEORB spans a whole day, but RESORB files (the fallback for recent
    S1C/S1D acquisitions, where the precise orbit still lags) are issued every
    orbit and overlap heavily, so several cover one scene. Among all covering
    candidates pick the one whose validity midpoint is closest to the
    acquisition midpoint -- the most centered orbit, giving the widest state-
    vector padding on both sides for isce3 ``geo2rdr`` at the scene edges.
    """
    fmt = "%Y%m%dT%H%M%S"
    mid = start + (stop - start) / 2
    best: str | None = None
    best_delta: float | None = None
    for name in names:
        m = _EOF_RE.match(name)
        if not m or m.group("mission") != mission:
            continue
        v_start = datetime.datetime.strptime(m.group("start"), fmt)
        v_stop = datetime.datetime.strptime(m.group("stop"), fmt)
        if v_start <= start and v_stop >= stop:
            v_mid = v_start + (v_stop - v_start) / 2
            delta = abs((v_mid - mid).total_seconds())
            if best_delta is None or delta < best_delta:
                best, best_delta = name, delta
    return best


def stage_orbit(
    granule: str, out_dir: Path, orbit_type: str = "auto", overwrite: bool = False
) -> Path:
    """Resolve and download the orbit covering ``granule`` from the ASF aux archive.

    Parameters
    ----------
    orbit_type :
        ``"POEORB"`` (precise), ``"RESORB"`` (restituted), or ``"auto"`` (try
        precise, fall back to restituted for acquisitions too recent for POEORB).
    """
    mission, start, stop = parse_granule(granule)
    session = earthdata_session()
    kinds = ["POEORB", "RESORB"] if orbit_type == "auto" else [orbit_type.upper()]

    for kind in kinds:
        match = _match_orbit(_list_orbits(session, kind), mission, start, stop)
        if match is None:
            print(f"  no {kind} covering {start:%Y-%m-%dT%H:%M:%S}")
            continue
        dest = out_dir / match
        if dest.exists() and not overwrite:
            print(f"  orbit present, skip: {dest.name}")
            return dest
        sub = "aux_poeorb" if kind == "POEORB" else "aux_resorb"
        print(f"  orbit ({kind}) download {match}")
        _stream_to_file(session, f"{ASF_AUX}/{sub}/{match}", dest)
        print(f"  -> {dest}")
        return dest

    sys.exit(f"No orbit found for {granule}")


# --------------------------------------------------------------------------- #
# 3. DEM (ellipsoidal GLO-30)
# --------------------------------------------------------------------------- #
def _footprint_bbox(granule: str) -> tuple[float, float, float, float]:
    """Return ``(w, s, e, n)`` for the granule from ASF's search API (no auth)."""
    from shapely import wkt

    resp = requests.get(
        ASF_SEARCH_API,
        params={"granule_list": _clean_name(granule), "output": "jsonlite"},
        timeout=60,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        sys.exit(f"ASF search returned no footprint for {granule}")
    minx, miny, maxx, maxy = wkt.loads(results[0]["wkt"]).bounds
    return (minx, miny, maxx, maxy)


def stage_dem(
    granule: str,
    out_dir: Path,
    margin_deg: float = 0.4,
    dem_name: str = "dem_4326.tiff",
    overwrite: bool = False,
    bbox: tuple[float, float, float, float] | None = None,
    snap_deg: float | None = None,
) -> Path:
    """Stitch an ellipsoidal Copernicus GLO-30 DEM covering the granule.

    The DEM must cover more than the SLC footprint: isce3 ``geo2rdr`` searches a
    height range at the scene edges, and each burst's geogrid (from the burst
    database) is padded, so a tight crop makes edge bursts warn ("limit may be
    insufficient") and can leave nodata. Hence a generous default ``margin_deg``.

    Parameters
    ----------
    margin_deg :
        Padding added around the granule footprint, in degrees. Default 0.4.
    bbox :
        Explicit ``(west, south, east, north)`` to stage instead of the
        footprint-derived box -- e.g. to match a golden dataset's DEM extent.
    snap_deg :
        If set, expand the (footprint+margin) box outward to a multiple of this
        many degrees (e.g. ``1.0`` snaps to whole-degree bounds, matching a DEM
        built on an integer-degree grid). Ignored when ``bbox`` is given.
    """
    import math

    import numpy as np
    import rasterio
    from dem_stitcher import stitch_dem

    dest = out_dir / dem_name
    if dest.exists() and not overwrite:
        print(f"  DEM present, skip: {dest.name}")
        return dest

    if bbox is not None:
        aoi = list(bbox)
    else:
        w, s, e, n = _footprint_bbox(granule)
        aoi = [w - margin_deg, s - margin_deg, e + margin_deg, n + margin_deg]
        if snap_deg:
            aoi = [
                math.floor(aoi[0] / snap_deg) * snap_deg,
                math.floor(aoi[1] / snap_deg) * snap_deg,
                math.ceil(aoi[2] / snap_deg) * snap_deg,
                math.ceil(aoi[3] / snap_deg) * snap_deg,
            ]
    print(f"  DEM stitch GLO-30 over {aoi}")
    array, profile = stitch_dem(
        aoi,
        "glo_30",
        dst_ellipsoidal_height=True,  # geoid -> WGS84 ellipsoid (isce3 expects this)
        dst_area_or_point="Point",
    )
    nodata = profile.get("nodata")
    if nodata is not None:
        array = np.where(array == nodata, 0.0, array)
    array = np.nan_to_num(array, nan=0.0).astype("float32")

    dest.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(profile)
    profile.update(
        driver="GTiff", dtype="float32", count=1, nodata=None, compress="DEFLATE"
    )
    with rasterio.open(dest, "w", **profile) as ds:
        ds.write(array, 1)
    print(f"  -> {dest} {array.shape}")
    return dest


# --------------------------------------------------------------------------- #
# 4. Ionosphere (IONEX TEC)
# --------------------------------------------------------------------------- #
def _ionex_candidates(
    date: datetime.date, sol_code: str, product_type: str, interval: str
) -> list[str]:
    """Return candidate CDDIS IONEX archive names for a day, preferred first.

    Covers both the legacy (``igrg1210.22i.Z``) and long IGS product
    (``IGS0OPSRAP_20221210000_01D_02H_GIM.INX.gz``) conventions.
    """
    doy = f"{date.timetuple().tm_yday:03d}"
    yy = f"{date.year % 100:02d}"
    legacy = f"{sol_code.lower()}g{doy}0.{yy}i.Z"
    new = f"{sol_code.upper()}0OPSFIN_{date.year}{doy}0000_01D_{interval}_GIM.INX.gz"
    if product_type.upper() == "RAPID":
        legacy = legacy.replace(sol_code.lower(), sol_code.lower()[:-1] + "r", 1)
        new = new.replace("0OPSFIN", "0OPSRAP")
    # Prefer the naming convention that matches the acquisition date.
    return [new, legacy] if date >= _NEW_IONEX_NAME_FROM else [legacy, new]


def stage_iono(
    granule: str,
    out_dir: Path,
    sol_code: str = "igs",
    product_type: str = "RAPID",
    interval: str = "02H",
    overwrite: bool = False,
) -> Path:
    """Download the daily IONEX TEC file for the granule date from NASA CDDIS.

    Defaults to the Rapid IGS (IGR) solution used by the CSLC-S1-SAS.
    """
    _, start, _ = parse_granule(granule)
    date = start.date()
    session = earthdata_session()

    doy = f"{date.timetuple().tm_yday:03d}"
    for archive in _ionex_candidates(date, sol_code, product_type, interval):
        url = f"{CDDIS_IONEX}/{date.year}/{doy}/{archive}"
        # Uncompressed target name (strip the .Z / .gz).
        uncompressed = out_dir / archive[: archive.rfind(".")]
        if uncompressed.exists() and not overwrite:
            print(f"  IONEX present, skip: {uncompressed.name}")
            return uncompressed
        resp = session.get(url, stream=True, timeout=120)
        if resp.status_code == 404:
            print(f"  not at {url}")
            continue
        resp.raise_for_status()

        out_dir.mkdir(parents=True, exist_ok=True)
        archive_path = out_dir / archive
        with open(archive_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)
        _decompress(archive_path, uncompressed)
        archive_path.unlink()
        print(f"  IONEX ({product_type} {sol_code}) -> {uncompressed}")
        return uncompressed

    sys.exit(
        f"No {product_type} {sol_code} IONEX at CDDIS for {date} "
        "(too recent, or try --product-type FINAL)"
    )


def _decompress(archive: Path, dest: Path) -> None:
    """Decompress a ``.gz`` (gzip) or legacy ``.Z`` (LZW) IONEX archive."""
    if archive.suffix == ".gz":
        with gzip.open(archive, "rb") as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:  # ".Z" is LZW; Python's gzip cannot read it, so shell out to gunzip.
        result = subprocess.run(
            ["gunzip", "-c", str(archive)], check=True, capture_output=True
        )
        dest.write_bytes(result.stdout)


# --------------------------------------------------------------------------- #
# 5. Burst database
# --------------------------------------------------------------------------- #
def stage_burst_db(
    out_dir: Path, source: str = DEFAULT_BURST_DB_URL, overwrite: bool = False
) -> Path:
    """Stage the OPERA burst-database SQLite from a URL or local path."""
    dest = out_dir / Path(urlparse(source).path if "://" in source else source).name
    if dest.exists() and not overwrite:
        print(f"  burst-db present, skip: {dest.name}")
        return dest

    out_dir.mkdir(parents=True, exist_ok=True)
    if "://" in source:
        print(f"  burst-db download {source}")
        _stream_to_file(requests.Session(), source, dest)
    else:
        print(f"  burst-db copy {source}")
        shutil.copy2(source, dest)
    print(f"  -> {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


# --------------------------------------------------------------------------- #
# 6. Runconfig
# --------------------------------------------------------------------------- #
_RUNCONFIG_TEMPLATE = """\
runconfig:
  name: cslc_s1_workflow_default

  groups:
      pge_name_group:
          pge_name: CSLC_S1_PGE

      input_file_group:
          safe_file_path:
          - {safe}
          orbit_file_path:
          - {orbit}
          burst_id:

      dynamic_ancillary_file_group:
          dem_file: {dem}
          tec_file: {tec}

      static_ancillary_file_group:
          burst_database_file: {burst_db}

      product_path_group:
          product_path: {product_path}
          scratch_path: {scratch_path}
          sas_output_file: {product_path}
          product_version:
          product_specification_version: {spec_version}

      primary_executable:
          product_type: {product_type}

      processing:
          polarization: co-pol
          geocoding:
              flatten: True
              x_posting: 5
              y_posting: 10
          geo2rdr:
              lines_per_block: 1000
              threshold: 1.0e-8
              numiter: 25

      worker:
          internet_access: False
          gpu_enabled: False
          gpu_id: 0
"""


def _find(out_dir: Path, pattern: str, kind: str) -> str:
    """Return the single ``input_dir``-relative match for ``pattern``."""
    matches = sorted(out_dir.glob(pattern))
    if not matches:
        sys.exit(f"Cannot write runconfig: no {kind} found ({pattern}). Stage it first.")
    return f"{out_dir.name}/{matches[0].name}"


def write_runconfig(
    granule: str,
    input_dir: Path,
    static: bool = False,
    work_dir: Path | None = None,
    tag: str | None = None,
) -> Path:
    """Write a runconfig referencing the staged inputs in ``input_dir``.

    ``input_dir`` is expected to be a child of ``work_dir`` (the run directory),
    matching the delivery layout where paths are ``input_data/<file>``.

    Parameters
    ----------
    tag :
        When several granules are staged under one run directory (each in its
        own ``input_dir``), pass a distinct tag so the runconfig filename and
        the product/scratch paths do not collide: ``runconfig_<tag>.yaml`` with
        ``output_<tag>`` / ``scratch_<tag>``. Defaults to the delivery names
        (``runconfig_cslc_s1.yaml``, ``output_s1_cslc``).
    """
    work_dir = work_dir or input_dir.parent
    suffix = "_static" if static else ""
    rc_base = tag or "cslc_s1"
    prod_base = tag or "s1_cslc"
    text = _RUNCONFIG_TEMPLATE.format(
        safe=_find(input_dir, "S1*_SLC*.zip", "SLC"),
        orbit=_find(input_dir, "S1*_AUX_*ORB_*.EOF", "orbit"),
        dem=_find(input_dir, "dem_*.tif*", "DEM"),
        tec=_find_tec(input_dir),
        burst_db=_find(input_dir, "*bbox-only.sqlite*", "burst database"),
        product_path=f"output_{prod_base}{suffix}",
        scratch_path=f"scratch_{prod_base}{suffix}",
        spec_version="0.1.2" if static else "0.1.7",
        product_type="CSLC_S1_STATIC" if static else "CSLC_S1",
    )
    dest = work_dir / f"runconfig_{rc_base}{suffix}.yaml"
    dest.write_text(text)
    print(f"  runconfig -> {dest}")
    return dest


def _find_tec(input_dir: Path) -> str:
    """Locate a staged IONEX file by either naming convention."""
    for pattern in ("*.??i", "*_GIM.INX", "*i"):
        matches = sorted(input_dir.glob(pattern))
        if matches:
            return f"{input_dir.name}/{matches[0].name}"
    sys.exit("Cannot write runconfig: no TEC/IONEX file found. Stage it first.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_common(parser: argparse.ArgumentParser, granule: bool = True) -> None:
    if granule:
        parser.add_argument("granule", help="Sentinel-1 SLC granule name.")
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=Path("input_data"),
        help="Directory to stage inputs into (default: input_data).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Restage even if present."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="stage_cslc_inputs.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_slc = sub.add_parser("slc", help="Download the SLC SAFE zip (ASF).")
    _add_common(p_slc)

    p_orb = sub.add_parser("orbit", help="Download the orbit EOF (ASF).")
    _add_common(p_orb)
    p_orb.add_argument(
        "--orbit-type",
        choices=["auto", "POEORB", "RESORB"],
        default="auto",
        help="Orbit type (default: auto = precise, else restituted).",
    )

    p_dem = sub.add_parser("dem", help="Stitch the ellipsoidal GLO-30 DEM.")
    _add_common(p_dem)
    p_dem.add_argument(
        "--margin", type=float, default=0.4, help="Footprint margin in degrees."
    )
    p_dem.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("W", "S", "E", "N"),
        default=None,
        help="Explicit DEM bounds instead of footprint+margin (e.g. to match a "
        "golden dataset).",
    )
    p_dem.add_argument(
        "--snap",
        type=float,
        default=None,
        help="Expand footprint+margin outward to a multiple of this many degrees "
        "(e.g. 1.0 for whole-degree bounds).",
    )

    p_ion = sub.add_parser("iono", help="Download the IONEX TEC file (CDDIS).")
    _add_common(p_ion)
    p_ion.add_argument("--sol-code", default="igs", help="IGS analysis center (igs=IGR).")
    p_ion.add_argument(
        "--product-type",
        choices=["RAPID", "FINAL"],
        default="RAPID",
        help="Rapid (IGR, SAS default) or Final solution.",
    )

    p_db = sub.add_parser("burst-db", help="Stage the burst-database SQLite.")
    _add_common(p_db, granule=False)
    p_db.add_argument(
        "--source", default=DEFAULT_BURST_DB_URL, help="URL or local path."
    )

    p_rc = sub.add_parser("runconfig", help="Write runconfig(s) from staged inputs.")
    _add_common(p_rc)
    p_rc.add_argument(
        "--static", action="store_true", help="Also emit the CSLC_S1_STATIC runconfig."
    )
    p_rc.add_argument(
        "--run-tag",
        default=None,
        help="Distinct tag for runconfig name and output/scratch paths "
        "(runconfig_<tag>.yaml, output_<tag>). Use when several granules share "
        "one run directory.",
    )

    p_all = sub.add_parser("all", help="Stage every input and write both runconfigs.")
    _add_common(p_all)
    p_all.add_argument("--margin", type=float, default=0.4)
    p_all.add_argument(
        "--dem-bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"), default=None,
        help="Explicit DEM bounds instead of footprint+margin.",
    )
    p_all.add_argument(
        "--dem-snap", type=float, default=None,
        help="Snap DEM footprint+margin outward to a multiple of this many degrees.",
    )
    p_all.add_argument("--sol-code", default="igs")
    p_all.add_argument("--product-type", choices=["RAPID", "FINAL"], default="RAPID")
    p_all.add_argument("--orbit-type", choices=["auto", "POEORB", "RESORB"], default="auto")
    p_all.add_argument("--burst-db-source", default=DEFAULT_BURST_DB_URL)
    p_all.add_argument(
        "--run-tag",
        default=None,
        help="Distinct tag for runconfig name and output/scratch paths "
        "(runconfig_<tag>.yaml, output_<tag>). Use when several granules share "
        "one run directory.",
    )

    args = parser.parse_args(argv)
    out = args.input_dir

    if args.cmd == "slc":
        stage_slc(args.granule, out, args.overwrite)
    elif args.cmd == "orbit":
        stage_orbit(args.granule, out, args.orbit_type, args.overwrite)
    elif args.cmd == "dem":
        stage_dem(
            args.granule, out, args.margin, overwrite=args.overwrite,
            bbox=tuple(args.bbox) if args.bbox else None, snap_deg=args.snap,
        )
    elif args.cmd == "iono":
        stage_iono(args.granule, out, args.sol_code, args.product_type, overwrite=args.overwrite)
    elif args.cmd == "burst-db":
        stage_burst_db(out, args.source, args.overwrite)
    elif args.cmd == "runconfig":
        write_runconfig(args.granule, out, static=False, tag=args.run_tag)
        if args.static:
            write_runconfig(args.granule, out, static=True, tag=args.run_tag)
    elif args.cmd == "all":
        print("[1/6] SLC")
        stage_slc(args.granule, out, args.overwrite)
        print("[2/6] orbit")
        stage_orbit(args.granule, out, args.orbit_type, args.overwrite)
        print("[3/6] DEM")
        stage_dem(
            args.granule, out, args.margin, overwrite=args.overwrite,
            bbox=tuple(args.dem_bbox) if args.dem_bbox else None, snap_deg=args.dem_snap,
        )
        print("[4/6] ionosphere")
        stage_iono(args.granule, out, args.sol_code, args.product_type, overwrite=args.overwrite)
        print("[5/6] burst database")
        stage_burst_db(out, args.burst_db_source, args.overwrite)
        print("[6/6] runconfigs")
        write_runconfig(args.granule, out, static=False, tag=args.run_tag)
        write_runconfig(args.granule, out, static=True, tag=args.run_tag)
        print("Done. All inputs staged under", out)


if __name__ == "__main__":
    main()

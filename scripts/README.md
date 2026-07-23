# CSLC-S1 staging and run scripts

- **`stage_cslc_inputs.py`** — stage all runconfig inputs for a granule (standalone).
- **`run_cslc.sh`** — stage, then run the CSLC-S1 SAS (`s1_cslc.py`) in the COMPASS
  Docker image on the generated runconfig.

## stage_cslc_inputs.py

Assembles every input the CSLC-S1 SAS runconfig needs into a local `input_data/`
directory and writes matching runconfig(s), mirroring the
`delivery_cslc_s1_*/input_data` layout.

It is **standalone**: it uses only packages already in the `compass` environment
(`requests`, `dem_stitcher`, `rasterio`/`gdal`, `shapely`) and adds no new
dependency. It does not import `compass`, `s1reader`, `sentineleof`, or
`asf_search`.

## Inputs staged

| Input | Source | Notes |
|-------|--------|-------|
| SLC SAFE zip | ASF DAAC HTTPS datapool | Earthdata Login |
| Orbit (EOF) | ASF S1 aux archive | by validity window; POEORB else RESORB |
| DEM (ellipsoidal) | Copernicus GLO-30 (`dem_stitcher`) | geoid → ellipsoid |
| Ionosphere (TEC) | NASA CDDIS | Rapid IGS (**IGR**), the SAS default |
| Burst database | OPERA `burst_db` release | public bbox-only SQLite |

## Credentials

ASF and CDDIS use NASA Earthdata Login. Add it once to `~/.netrc`:

```text
machine urs.earthdata.nasa.gov login <user> password <pass>
```

(or export `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`).

## Usage

Stage everything for one granule and write both runconfigs:

```bash
python scripts/stage_cslc_inputs.py all \
    S1A_IW_SLC__1SDV_20220501T015035_20220501T015102_043011_0522A4_42CC
```

Run from the directory where you want `input_data/` and the runconfigs created.

### Downloading the SLC (SAFE)

Fetch just the SLC SAFE zip for a granule (Earthdata Login required):

```bash
python scripts/stage_cslc_inputs.py slc \
    S1A_IW_SLC__1SDV_20220501T015035_20220501T015102_043011_0522A4_42CC
```

This streams the full `<GRANULE>.zip` from ASF's HTTPS datapool
(`https://datapool.asf.alaska.edu/SLC/S{A,C,D}/<GRANULE>.zip`) into the
`-i/--input-dir` (default `input_data/`). The mission letter (`A`/`C`/`D`) is
taken from the granule name, so S1A/S1C/S1D are all supported. A granule already
present is skipped unless you pass `--overwrite`.

> **One runconfig per SAFE.** A runconfig references exactly one SLC. With a
> single `S1*_SLC*.zip` staged you get one runconfig (default names
> `runconfig_cslc_s1.yaml` / `output_s1_cslc`). Stage several SAFEs of a stack
> into the same directory and `runconfig`/`all` writes **one runconfig per
> SAFE**, matching each to its own orbit (by validity window) and TEC (by date)
> and tagging it by acquisition date (`runconfig_20220501.yaml` /
> `output_20220501`, …). DEM and burst-db are shared across all of them.

### Individual steps

```bash
python scripts/stage_cslc_inputs.py slc <GRANULE>
python scripts/stage_cslc_inputs.py orbit <GRANULE> [--orbit-type POEORB|RESORB]
python scripts/stage_cslc_inputs.py dem <GRANULE> \
    [--margin 0.4] [--bbox W S E N] [--snap 1.0]
python scripts/stage_cslc_inputs.py iono <GRANULE> \
    [--product-type RAPID|FINAL] [--sol-code igs]
python scripts/stage_cslc_inputs.py burst-db [--source URL|PATH]
python scripts/stage_cslc_inputs.py runconfig <GRANULE> [--static] [--run-tag TAG]
```

Every step takes `-i/--input-dir` (default `input_data`) and `--overwrite`.
Downloads skip files already present, so reruns only fetch the gaps.

**Ionosphere** defaults to the **Rapid IGS (IGR)** solution the CSLC-S1-SAS uses;
pass `--product-type FINAL --sol-code jpl` for the final `jplg`-style file.

**DEM buffer.** The DEM must cover more than the SLC footprint (isce3 `geo2rdr`
searches a height range at the edges, and each burst's geogrid is padded), so
`--margin` defaults to 0.4 deg. Increase it to match a golden delivery's extent
(e.g. `--margin 1.7`), or set an explicit `--bbox W S E N`.

**Staging a stack (several granules, one area).** DEM and burst-db are area- and
mission-independent, so one copy serves every granule over the same area. Stage
each date's SLC/orbit/TEC into a single shared `input_data/` (repeat the `slc` /
`orbit` / `iono` steps per granule, or run `all` once per granule into the same
`-i` dir), then run `runconfig` once: it emits one runconfig per SAFE, each
matched to its own orbit/TEC and tagged by date, all pointing at the shared
`dem_4326.tiff` and burst-db. To keep granules fully separate instead, stage
each into its own `-i input_data_<tag>` dir with `--run-tag <tag>`.

## run_cslc.sh

Stages inputs (step 1) then runs the SAS in Docker (step 2). The run directory is
bind-mounted into the container at the same path, so the relative paths in the
runconfig resolve unchanged.

```bash
# stage + run for one granule (workdir defaults to ./<GRANULE>)
scripts/run_cslc.sh <GRANULE> --workdir /path/to/run

# reproduce a golden delivery (final-jpl TEC, wider DEM buffer)
scripts/run_cslc.sh <GRANULE> -- --product-type FINAL --sol-code jpl --margin 1.7

# reuse already-staged inputs, or only stage
scripts/run_cslc.sh <GRANULE> --workdir DIR --skip-staging
scripts/run_cslc.sh <GRANULE> --workdir DIR --skip-sas
```

Anything after `--` is forwarded to `stage_cslc_inputs.py`. Options: `--image`
(default `opera/cslc_s1:final_0.5.7`), `--python`, `--no-user`, `--docker-arg`.
Requires the COMPASS Docker image built (`./build_docker_image.sh`) and, for
S1C/S1D, s1-reader >= v0.2.6 in that image (the 2026-06-24 S1C maneuver).

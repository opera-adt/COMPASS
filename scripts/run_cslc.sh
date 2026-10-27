#!/usr/bin/env bash
# Stage all CSLC-S1 inputs for a Sentinel-1 granule, then run the CSLC-S1 SAS
# (s1_cslc.py) inside the COMPASS Docker image on the generated runconfig.
#
# Two steps, either of which can be skipped:
#   1. stage_cslc_inputs.py all <granule> -i <workdir>/input_data   (host, needs
#      Earthdata ~/.netrc; see scripts/README.md)
#   2. docker run ... s1_cslc.py <workdir>/runconfig_cslc_s1.yaml
#
# The workdir is bind-mounted into the container at the SAME path and used as the
# working directory, so the relative paths the stager writes into the runconfig
# (input_data/..., output_s1_cslc) resolve unchanged inside the container.
#
# Usage:
#   run_cslc.sh <GRANULE> [options] [-- <extra stage_cslc_inputs.py args>]
#
# Options:
#   --workdir DIR         Run directory (default: ./<GRANULE>).
#   --image IMAGE         Docker image (default: opera/cslc_s1:final_0.5.7).
#   --python PY           Python for staging (default: python).
#   --az-timing MODE      Azimuth timing correction: none, carrier_phase or
#                         geometry. Offsets come from the timing table (the one
#                         packaged with COMPASS unless --timing-table is given),
#                         or from --az-time-offset.
#   --range-timing MODE   Range timing correction: none or geometry.
#   --timing-table FILE   OPERA_S1_AUX_TIMING table (copied into the workdir so
#                         the container sees it).
#   --az-time-offset SEC  Constant azimuth-time offset (seconds) for every burst
#                         instead of the table. Without --az-timing it is a
#                         geometry shift.
#   (The timing options need an image built from a COMPASS with
#    processing.timing_correction.)
#   --skip-staging        Reuse inputs already in <workdir>/input_data.
#   --skip-sas            Only stage; do not run the SAS.
#   --no-user             Run the container as its default user (do not map host uid).
#   --docker-arg ARG      Extra `docker run` arg (repeatable).
#   -h, --help            This help.
#
# Everything after `--` is forwarded to stage_cslc_inputs.py, e.g. to match a
# golden delivery:
#   run_cslc.sh <GRANULE> -- --product-type FINAL --sol-code jpl --margin 1.7
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGER="$SCRIPT_DIR/stage_cslc_inputs.py"

IMAGE="opera/cslc_s1:final_0.5.7"
PYTHON="${PYTHON:-python}"
WORKDIR=""
RUN_STAGING=1
RUN_SAS=1
MAP_USER=1
AZ_OFFSET=""
AZ_TIMING=""
RANGE_TIMING=""
TIMING_TABLE=""
declare -a STAGE_ARGS=()
declare -a DOCKER_ARGS=()
GRANULE=""

usage() { sed -n '2,46p' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workdir)    WORKDIR="$2"; shift 2;;
        --image)      IMAGE="$2"; shift 2;;
        --python)     PYTHON="$2"; shift 2;;
        --az-time-offset) AZ_OFFSET="$2"; shift 2;;
        --az-timing)  AZ_TIMING="$2"; shift 2;;
        --range-timing) RANGE_TIMING="$2"; shift 2;;
        --timing-table) TIMING_TABLE="$2"; shift 2;;
        --skip-staging) RUN_STAGING=0; shift;;
        --skip-sas)   RUN_SAS=0; shift;;
        --no-user)    MAP_USER=0; shift;;
        --docker-arg) DOCKER_ARGS+=("$2"); shift 2;;
        --)           shift; STAGE_ARGS+=("$@"); break;;
        -h|--help)    usage; exit 0;;
        -*)           echo "Unknown option: $1" >&2; usage; exit 1;;
        *)            GRANULE="$1"; shift;;
    esac
done

[[ -n "$GRANULE" ]] || { echo "ERROR: no granule given" >&2; usage; exit 1; }
GRANULE="${GRANULE%.zip}"; GRANULE="${GRANULE%.SAFE}"
: "${WORKDIR:=$PWD/$GRANULE}"
mkdir -p "$WORKDIR"
WORKDIR="$(cd "$WORKDIR" && pwd)"
INPUT_DIR="$WORKDIR/input_data"
RUNCONFIG="$WORKDIR/runconfig_cslc_s1.yaml"

if [[ "$RUN_STAGING" -eq 1 ]]; then
    echo ">>> [1/2] Staging inputs for $GRANULE -> $INPUT_DIR"
    "$PYTHON" "$STAGER" all "$GRANULE" -i "$INPUT_DIR" "${STAGE_ARGS[@]}"
else
    echo ">>> [1/2] Skipping staging (reusing $INPUT_DIR)"
fi

if [[ "$RUN_SAS" -eq 0 ]]; then
    echo ">>> [2/2] Skipping SAS (--skip-sas). Runconfig: $RUNCONFIG"
    exit 0
fi

[[ -f "$RUNCONFIG" ]] || { echo "ERROR: runconfig not found: $RUNCONFIG" >&2; exit 1; }

# Constant offset without a mode keeps the old meaning: a geometry shift
if [[ -n "$AZ_OFFSET" && -z "$AZ_TIMING" ]]; then
    AZ_TIMING=geometry
fi
if [[ -n "$TIMING_TABLE" ]]; then
    cp "$TIMING_TABLE" "$WORKDIR/"
    TIMING_TABLE="$WORKDIR/$(basename "$TIMING_TABLE")"
fi

if [[ -n "$AZ_TIMING$RANGE_TIMING$TIMING_TABLE" ]]; then
    echo ">>> Setting processing.timing_correction in $RUNCONFIG" \
         "(azimuth=${AZ_TIMING:-none}, range=${RANGE_TIMING:-none}," \
         "offset=${AZ_OFFSET:-table}, table=${TIMING_TABLE:-packaged})"
    "$PYTHON" - "$RUNCONFIG" "${AZ_TIMING:-none}" "${RANGE_TIMING:-none}" \
        "$AZ_OFFSET" "$TIMING_TABLE" <<'EOF'
# Replace any existing timing_correction block and timing_table_file line
# (idempotent under --skip-staging reruns) and insert fresh ones.
import re
import sys

path, az_mode, rg_mode, az_offset, table = sys.argv[1:6]
text = open(path).read()
text = re.sub(r"\n {10}timing_correction:\n(?: {12,}.*\n)*", "\n", text)
text = re.sub(r" {10}timing_table_file:.*\n", "", text)
block = (
    "          timing_correction:\n"
    f"              azimuth_mode: {az_mode}\n"
    f"              range_mode: {rg_mode}\n"
)
if az_offset:
    block += ("              offset_source: constant\n"
              f"              azimuth_constant_offset: {az_offset}\n")
text = text.replace("      worker:", block + "\n      worker:", 1)
if table:
    text = text.replace(
        "      static_ancillary_file_group:\n",
        "      static_ancillary_file_group:\n"
        f"          timing_table_file: {table}\n", 1)
open(path, "w").write(text)
EOF
fi

run_args=(--rm -v "$WORKDIR":"$WORKDIR" -w "$WORKDIR")
if [[ "$MAP_USER" -eq 1 ]]; then
    # Map the host user so products are host-owned; HOME must be writable for the
    # numba/isce3 caches when not running as the image's built-in user.
    run_args+=(-u "$(id -u):$(id -g)" -e HOME=/tmp)
fi
run_args+=("${DOCKER_ARGS[@]}" "$IMAGE" s1_cslc.py "$RUNCONFIG")

echo ">>> [2/2] Running CSLC-S1 SAS in Docker: $IMAGE"
echo "    docker run ${run_args[*]}"
docker run "${run_args[@]}"

echo ">>> Done. Products under $WORKDIR/output_s1_cslc"

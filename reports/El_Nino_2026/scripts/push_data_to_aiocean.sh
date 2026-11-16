#!/usr/bin/env bash
# Push the data downloaded for the Line 66.7 El Niño report to the AIOcean
# Google Drive (rclone remote "AIOcean").  Safe to re-run: rclone copy only
# uploads new or changed files and never deletes anything on the remote.
#
# Remote layout (under AIOcean:data/Spray/):
#   CUGN/Line_66/                 glider data: binnedCUGN66.nc, DAC/, products/
#   El_Nino_2026/CCS/             ONI, SCTI, CUTI/BEUTI, CO-OPS tide gauges ($OS_CCS)
#   El_Nino_2026/SST/OISST/       OISST v2.1 yearly CCS subsets (cugn.oisst)
#
# Usage:  bash reports/El_Nino_2026/scripts/push_data_to_aiocean.sh [--dry-run]
# Written by JXP and Claude, 2026-09-12.
set -euo pipefail

REMOTE=AIOcean:data/Spray
OS_SPRAY=${OS_SPRAY:-/home/xavier/Oceanography/data/Spray/}
OS_CCS=${OS_CCS:-/home/xavier/Oceanography/data/CCS/}
OISST_DIR=$(dirname "${OS_SPRAY%/}")/SST/OISST

OPTS=(--drive-chunk-size 64M --transfers 4 --checkers 8 --stats-one-line -v "$@")

# The SprayData dump is re-downloaded every month but its data only change
# when a mission's delayed-mode QC is finished; ERDDAP rewrites the header on
# every request, so the file never matches byte-for-byte.  Compare a hash of
# the data with the one stored on the remote and skip the ~450 MB upload when
# they agree (JXP, Oct 2026).
LINE66="${OS_SPRAY%/}/CUGN/Line_66"
HASHFILE=binnedCUGN66.datahash
EXCLUDE=()
if [ -f "$LINE66/binnedCUGN66.nc" ]; then
    local_hash=$(conda run -n ocean14 --no-capture-output python -c \
        "from cugn import erddap; print(erddap.table_hash('$LINE66/binnedCUGN66.nc'))")
    remote_hash=$(rclone cat "$REMOTE/CUGN/Line_66/$HASHFILE" 2>/dev/null || true)
    if [ -n "$local_hash" ] && [ "$local_hash" = "$remote_hash" ]; then
        echo "== binnedCUGN66.nc: data unchanged since the last push, skipping the dump"
        EXCLUDE=(--exclude binnedCUGN66.nc)
    else
        echo "$local_hash" > "$LINE66/$HASHFILE"
    fi
fi

echo "== glider data -> $REMOTE/CUGN/Line_66"
rclone copy "$LINE66" "$REMOTE/CUGN/Line_66" "${EXCLUDE[@]}" "${OPTS[@]}"

echo "== CCS indices -> $REMOTE/El_Nino_2026/CCS"
rclone copy "${OS_CCS%/}" "$REMOTE/El_Nino_2026/CCS" "${OPTS[@]}"

echo "== OISST -> $REMOTE/El_Nino_2026/SST/OISST"
rclone copy "$OISST_DIR" "$REMOTE/El_Nino_2026/SST/OISST" \
    --include 'oisst_ccs_*.nc' "${OPTS[@]}"

echo "== verify"
rclone check "$LINE66" "$REMOTE/CUGN/Line_66" --one-way --size-only "${EXCLUDE[@]}"
rclone check "${OS_CCS%/}" "$REMOTE/El_Nino_2026/CCS" --one-way --size-only
rclone check "$OISST_DIR" "$REMOTE/El_Nino_2026/SST/OISST" --one-way --size-only --include 'oisst_ccs_*.nc'
echo "done"

#!/usr/bin/env bash
set -euo pipefail

ENV_PY="${ENV_PY:-/root/miniconda3/envs/puma/bin/python}"
CONFIG="${CONFIG:-config.json}"
THREADS="${THREADS:-4}"

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"

mkdir -p logs

echo "[whole-roi] starting WM with THREADS=$THREADS"
"$ENV_PY" scripts/06_run_puma_whole_roi.py --config "$CONFIG" --sample WM \
  > logs/06_whole_roi_WM.log 2>&1

echo "[whole-roi] starting PFDT with THREADS=$THREADS"
"$ENV_PY" scripts/06_run_puma_whole_roi.py --config "$CONFIG" --sample PFDT \
  > logs/06_whole_roi_PFDT.log 2>&1

"$ENV_PY" - <<'PY'
from pathlib import Path
import pandas as pd

base = Path("results")
frames = []
for sample in ["WM", "PFDT"]:
    frames.append(pd.read_csv(base / sample / "whole_roi_transport.csv"))
pd.concat(frames, ignore_index=True).to_csv(base / "whole_roi_transport.csv", index=False)
print(f"Saved {base / 'whole_roi_transport.csv'}")
PY

echo "[whole-roi] complete"

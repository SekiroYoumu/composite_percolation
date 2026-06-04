#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-config.json}"

echo "Run this inside tmux/screen on the server for long jobs."
echo "Using Python: ${PYTHON_BIN}"
echo "Using config: ${CONFIG}"

"${PYTHON_BIN}" scripts/01_validate_and_merge_masks.py --config "${CONFIG}" --skip-existing 2>&1 | tee logs/01_validate_and_merge_masks.log
"${PYTHON_BIN}" scripts/02_puma_smoke_test.py --config "${CONFIG}" 2>&1 | tee logs/02_puma_smoke_test.log
"${PYTHON_BIN}" scripts/03_generate_subvolumes.py --config "${CONFIG}" --skip-existing 2>&1 | tee logs/03_generate_subvolumes.log
"${PYTHON_BIN}" scripts/04_run_puma_batch.py --config "${CONFIG}" 2>&1 | tee logs/04_run_puma_batch.log
"${PYTHON_BIN}" scripts/05_plot_results.py --config "${CONFIG}" 2>&1 | tee logs/05_plot_results.log

echo "Pipeline finished. Outputs are under results/."

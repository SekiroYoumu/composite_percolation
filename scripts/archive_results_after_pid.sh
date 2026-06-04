#!/usr/bin/env bash
set -euo pipefail

WATCH_PID="${1:?usage: archive_results_after_pid.sh <pid-to-watch> <archive-dir>}"
ARCHIVE_DIR="${2:?usage: archive_results_after_pid.sh <pid-to-watch> <archive-dir>}"

while kill -0 "$WATCH_PID" 2>/dev/null; do
  sleep 120
done

mkdir -p "$ARCHIVE_DIR"
cp -a results "$ARCHIVE_DIR/"
cp -a logs "$ARCHIVE_DIR/" 2>/dev/null || true
cp -a config.json config_representative_recompute.json "$ARCHIVE_DIR/" 2>/dev/null || true

{
  echo "archive_name=$(basename "$ARCHIVE_DIR")"
  echo "created_after_pid=$WATCH_PID"
  echo "representative_source=transport_results.csv median selection"
  echo
  echo "disk_usage:"
  du -sh "$ARCHIVE_DIR" results 2>/dev/null || true
} > "$ARCHIVE_DIR/MANIFEST.txt"

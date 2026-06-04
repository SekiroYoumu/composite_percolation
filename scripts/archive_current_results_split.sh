#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

ARCHIVE_ROOT="${ARCHIVE_ROOT:-archive}"
THIRTY_DIR="$ARCHIVE_ROOT/final_20260604_30um_edge_y_representative_fields"
WHOLE_DIR="$ARCHIVE_ROOT/final_20260604_whole_roi_y_downsample2"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -e "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$dst"
  fi
}

copy_matching_files() {
  local src_dir="$1"
  local dst_dir="$2"
  shift 2
  mkdir -p "$dst_dir"
  for pattern in "$@"; do
    find "$src_dir" -maxdepth 1 -type f -name "$pattern" -exec cp -a {} "$dst_dir/" \; 2>/dev/null || true
  done
}

rm -rf "$THIRTY_DIR" "$WHOLE_DIR"
mkdir -p "$THIRTY_DIR" "$WHOLE_DIR"

# 30 um sub-volume statistics and representative fields.
for sample in WM PFDT; do
  copy_if_exists "results/$sample/subvolumes.csv" "$THIRTY_DIR/results/$sample/subvolumes.csv"
  copy_if_exists "results/$sample/transport_results.csv" "$THIRTY_DIR/results/$sample/transport_results.csv"
  copy_if_exists "results/$sample/label_preview_slices.png" "$THIRTY_DIR/results/$sample/label_preview_slices.png"
  if [[ -d "results/$sample/representative_flux" ]]; then
    mkdir -p "$THIRTY_DIR/results/$sample/representative_flux"
    cp -a results/$sample/representative_flux/* "$THIRTY_DIR/results/$sample/representative_flux/"
  fi
done
copy_if_exists "results/volume_fractions.csv" "$THIRTY_DIR/results/volume_fractions.csv"
copy_matching_files "results/figures" "$THIRTY_DIR/results/figures" \
  "summary.csv" \
  "keff_norm_scatter.png" \
  "phase_fraction_scatter.png" \
  "concentration_linear.png" \
  "flux_magnitude_linear.png" \
  "flux_magnitude_log.png" \
  "representative_*.png"
copy_matching_files "logs" "$THIRTY_DIR/logs" \
  "01_validate_30um_edge_y.log" \
  "03_generate_30um_edge_y.log" \
  "04_run_puma_batch_30um_edge_y*.log" \
  "04_recompute_30um_representative_fields.log" \
  "05_plot_results_after_30um_edge_y*.log" \
  "07_plot_representative_orthogonal_slices.log" \
  "pumapy_log_2026-06-03*.txt" \
  "pumapy_log_2026-06-04_06;49;22.txt" \
  "pumapy_log_2026-06-04_08;40;32.txt"

# Whole-ROI scalar result and downsampled 3D fields.
for sample in WM PFDT; do
  copy_if_exists "results/$sample/whole_roi_transport.csv" "$WHOLE_DIR/results/$sample/whole_roi_transport.csv"
  if [[ -d "bulk_fields/current/$sample/fields/whole_roi" ]]; then
    mkdir -p "$WHOLE_DIR/bulk_fields/current/$sample/fields"
    cp -a "bulk_fields/current/$sample/fields/whole_roi" "$WHOLE_DIR/bulk_fields/current/$sample/fields/"
  fi
done
copy_if_exists "results/whole_roi_transport.csv" "$WHOLE_DIR/results/whole_roi_transport.csv"
copy_matching_files "results/figures" "$WHOLE_DIR/results/figures" "whole_roi_*.png"
copy_matching_files "logs" "$WHOLE_DIR/logs" \
  "06_whole_roi*.log" \
  "08_plot_whole_roi_downsampled_comparison.log" \
  "run_whole_roi_server.log" \
  "whole_roi_summary_watcher.log" \
  "pumapy_log_2026-06-04_06;39;07.txt" \
  "pumapy_log_2026-06-04_10;00;27.txt" \
  "pumapy_log_2026-06-04_10;46;03.txt"

for archive_dir in "$THIRTY_DIR" "$WHOLE_DIR"; do
  copy_if_exists "config.json" "$archive_dir/config.json"
  copy_if_exists "README.md" "$archive_dir/README.md"
  if [[ -d scripts ]]; then
    mkdir -p "$archive_dir/scripts"
    cp -a scripts/*.py scripts/*.sh "$archive_dir/scripts/" 2>/dev/null || true
  fi
done

cat > "$THIRTY_DIR/MANIFEST.txt" <<'EOF'
archive_name=final_20260604_30um_edge_y_representative_fields
calculation=30 um edge-aligned subvolumes, direction y
contents=subvolume CSVs, transport_results.csv, representative potential/flux fields, representative comparison figures
note=label_zyx.tif is excluded because it is an intermediate file
EOF

cat > "$WHOLE_DIR/MANIFEST.txt" <<'EOF'
archive_name=final_20260604_whole_roi_y_downsample2
calculation=whole ROI, direction y
contents=whole_roi_transport.csv, 2x float32 downsampled potential/concentration and flux_magnitude fields, whole-ROI comparison figures
note=label_zyx.tif is excluded because it is an intermediate file
EOF

{
  echo
  echo "disk_usage:"
  du -sh "$THIRTY_DIR" "$THIRTY_DIR/results" "$THIRTY_DIR/logs" 2>/dev/null || true
} >> "$THIRTY_DIR/MANIFEST.txt"

{
  echo
  echo "disk_usage:"
  du -sh "$WHOLE_DIR" "$WHOLE_DIR/results" "$WHOLE_DIR/bulk_fields" "$WHOLE_DIR/logs" 2>/dev/null || true
} >> "$WHOLE_DIR/MANIFEST.txt"

echo "Created:"
du -sh "$THIRTY_DIR" "$WHOLE_DIR"

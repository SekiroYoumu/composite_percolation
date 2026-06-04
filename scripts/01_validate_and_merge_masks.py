from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline_common import (
    compact_unique,
    input_path,
    label_fractions,
    load_config,
    read_volume_as_zyx,
    sample_label_path,
    sample_results_dir,
    save_label_preview,
    write_label_tiff,
    write_rows_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate binary phase masks and merge them into integer label volumes.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a sample when its label_zyx.tif already exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    labels = cfg["labels"]
    all_rows = []

    for sample_key, sample in cfg["samples"].items():
        out_dir = sample_results_dir(cfg, sample_key)
        existing_label_path = sample_label_path(cfg, sample_key)
        label_path = sample_label_path(cfg, sample_key, prefer_existing=False)
        if args.skip_existing and existing_label_path.exists():
            print(f"[{sample_key}] label exists, skipping merge: {existing_label_path}")
            continue

        print(f"\n[{sample_key}] {sample.get('name', sample_key)}")
        masks_bool: dict[str, np.ndarray] = {}
        shapes = {}
        for phase in ("CAM", "SE-rich", "Void/carbon-rich"):
            path = input_path(cfg, sample["masks"][phase])
            array = read_volume_as_zyx(Path(path), cfg.get("tiff_axis_order", "z_y_x"))
            print(
                f"  {phase:18s} path={path.name} shape={array.shape} dtype={array.dtype} "
                f"min={array.min()} max={array.max()} unique={compact_unique(array)}"
            )
            masks_bool[phase] = array > 0
            shapes[phase] = array.shape

        if len(set(shapes.values())) != 1:
            raise ValueError(f"[{sample_key}] Mask shapes are inconsistent: {shapes}")

        cam = masks_bool["CAM"]
        se = masks_bool["SE-rich"]
        void = masks_bool["Void/carbon-rich"]
        occupancy = cam.astype(np.uint8) + se.astype(np.uint8) + void.astype(np.uint8)
        overlap = int(np.count_nonzero(occupancy > 1))
        unassigned = int(np.count_nonzero(occupancy == 0))
        total = int(occupancy.size)
        print(f"  overlap voxels:   {overlap} ({overlap / total:.6g})")
        print(f"  unassigned voxels:{unassigned} ({unassigned / total:.6g})")

        if overlap:
            print("  WARNING: overlap exists; priority is CAM > SE-rich > Void/carbon-rich.")
        if unassigned and not cfg.get("fill_unassigned_as_void", False):
            print("  WARNING: unassigned voxels remain label 0. Set fill_unassigned_as_void=true to map them to 3.")

        label = np.zeros(next(iter(shapes.values())), dtype=np.uint8)
        label[void] = int(labels["Void/carbon-rich"])
        label[se] = int(labels["SE-rich"])
        label[cam] = int(labels["CAM"])
        if cfg.get("fill_unassigned_as_void", False):
            label[label == 0] = int(labels["Void/carbon-rich"])

        write_label_tiff(label_path, label)
        print(f"  saved label volume: {label_path}")

        fractions = label_fractions(label, labels)
        row = {"sample": sample_key, "name": sample.get("name", sample_key), "shape_zyx": "x".join(map(str, label.shape))}
        row.update(fractions)
        all_rows.append(row)
        for key, value in fractions.items():
            print(f"  {key}: {value:.6f}")

        preview_path = out_dir / "label_preview_slices.png"
        save_label_preview(label, preview_path, labels, sample_key)
        print(f"  saved preview PNG: {preview_path}")

    if all_rows:
        summary_path = Path(cfg["_root"]) / cfg.get("results_dir", "results") / "volume_fractions.csv"
        write_rows_csv(summary_path, all_rows)
        print(f"\nSaved volume fraction summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

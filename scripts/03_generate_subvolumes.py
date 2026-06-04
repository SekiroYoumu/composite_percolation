from __future__ import annotations

import argparse

import numpy as np

from pipeline_common import (
    label_fractions,
    load_config,
    read_volume_as_zyx,
    sample_label_path,
    sample_results_dir,
    um_size_to_zyx_voxels,
    write_rows_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate regular grid or sliding-window sub-volume coordinates.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples with existing subvolumes.csv.")
    parser.add_argument("--subvolume-um", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Override sub-volume size in um.")
    parser.add_argument("--stride-um", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Override sliding-window stride in um.")
    return parser.parse_args()


def starts_for_full_windows(dim: int, size: int, stride: int) -> list[int]:
    if dim < size:
        return []
    starts = list(range(0, dim - size + 1, stride))
    return starts


def edge_aligned_starts(dim: int, size: int, count: int) -> list[int]:
    if dim < size:
        return []
    if count <= 1:
        return [(dim - size) // 2]
    max_start = dim - size
    starts = [int(round(i * max_start / (count - 1))) for i in range(count)]
    return sorted(set(starts))


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    labels = cfg["labels"]
    subvolume_um = args.subvolume_um if args.subvolume_um else cfg.get("subvolume_um", [25, 25, 25])
    stride_um = args.stride_um if args.stride_um else cfg.get("stride_um")
    sampling_mode = str(cfg.get("sampling_mode", "grid")).lower()
    if stride_um is None:
        stride_um = subvolume_um
    sub_zyx = um_size_to_zyx_voxels(subvolume_um, cfg)
    stride_zyx = um_size_to_zyx_voxels(stride_um, cfg)
    print(f"Sub-volume size: {subvolume_um} um -> shape_zyx={sub_zyx}")
    print(f"Sampling mode: {sampling_mode}")
    print(f"Stride: {stride_um} um -> stride_zyx={stride_zyx}")

    for sample_key in cfg["samples"]:
        out_dir = sample_results_dir(cfg, sample_key)
        csv_path = out_dir / "subvolumes.csv"
        if args.skip_existing and csv_path.exists():
            print(f"[{sample_key}] subvolumes.csv exists, skipping: {csv_path}")
            continue

        label_path = sample_label_path(cfg, sample_key)
        if not label_path.exists():
            raise FileNotFoundError(f"Missing {label_path}; run 01_validate_and_merge_masks.py first.")
        label = read_volume_as_zyx(label_path, "z_y_x")
        zmax, ymax, xmax = label.shape
        sz, sy, sx = sub_zyx
        dz, dy, dx = stride_zyx
        if sampling_mode == "edge_aligned":
            counts = cfg.get("edge_aligned_counts", {"x": 1, "y": 2, "z": 2})
            x_starts = edge_aligned_starts(xmax, sx, int(counts.get("x", 1)))
            y_starts = edge_aligned_starts(ymax, sy, int(counts.get("y", 2)))
            z_starts = edge_aligned_starts(zmax, sz, int(counts.get("z", 2)))
        else:
            z_starts = starts_for_full_windows(zmax, sz, dz)
            y_starts = starts_for_full_windows(ymax, sy, dy)
            x_starts = starts_for_full_windows(xmax, sx, dx)
        if not z_starts or not y_starts or not x_starts:
            raise ValueError(f"[{sample_key}] ROI shape {label.shape} is smaller than subvolume shape {sub_zyx}.")

        rows = []
        sub_id = 0
        for z0 in z_starts:
            z1 = z0 + sz
            for y0 in y_starts:
                y1 = y0 + sy
                for x0 in x_starts:
                    x1 = x0 + sx
                    block = label[z0:z1, y0:y1, x0:x1]
                    sub_id += 1
                    row = {
                        "sample": sample_key,
                        "subvolume_id": f"{sample_key}_SV{sub_id:04d}",
                        "x0": x0,
                        "x1": x1,
                        "y0": y0,
                        "y1": y1,
                        "z0": z0,
                        "z1": z1,
                        "shape_x": sx,
                        "shape_y": sy,
                        "shape_z": sz,
                        "stride_x": dx,
                        "stride_y": dy,
                        "stride_z": dz,
                        "sampling_mode": sampling_mode,
                    }
                    row.update(label_fractions(block, labels))
                    rows.append(row)

        fieldnames = [
            "sample",
            "subvolume_id",
            "x0",
            "x1",
            "y0",
            "y1",
            "z0",
            "z1",
            "shape_x",
            "shape_y",
            "shape_z",
            "stride_x",
            "stride_y",
            "stride_z",
            "sampling_mode",
            "CAM fraction",
            "SE-rich fraction",
            "Void/carbon-rich fraction",
            "unassigned fraction",
        ]
        write_rows_csv(csv_path, rows, fieldnames)
        last_end = np.array([z_starts[-1] + sz, y_starts[-1] + sy, x_starts[-1] + sx], dtype=int)
        ignored = np.array(label.shape, dtype=int) - last_end
        print(f"[{sample_key}] ROI shape_zyx={label.shape}; grid x*y*z={len(x_starts)}*{len(y_starts)}*{len(z_starts)}; n={len(rows)}")
        print(f"[{sample_key}] ignored edge voxels z,y,x={tuple(ignored)}; saved {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

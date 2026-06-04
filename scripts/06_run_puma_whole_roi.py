from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from pipeline_common import (
    compute_transport,
    current_memory_mb,
    label_fractions,
    load_config,
    read_volume_as_zyx,
    sample_fields_dir,
    sample_label_path,
    sample_results_dir,
    save_scalar_orthogonal_slices,
    write_rows_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PuMA transport calculation on the full label volume.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--sample", choices=["WM", "PFDT"], help="Run only one sample.")
    parser.add_argument("--force", action="store_true", help="Recompute even if a whole-ROI result already exists.")
    return parser.parse_args()


def result_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if df.empty or "keff_norm" not in df or "status" not in df:
        return False
    return bool((df["status"].astype(str) == "ok").all() and np.isfinite(df["keff_norm"].astype(float)).all())


def downsample_mean(array: np.ndarray, factor: int, dtype: str = "float32") -> np.ndarray:
    arr = np.asarray(array)
    if factor <= 1:
        return arr.astype(dtype, copy=False)
    cropped_shape = tuple((dim // factor) * factor for dim in arr.shape)
    slices = tuple(slice(0, dim) for dim in cropped_shape)
    arr = arr[slices].astype(dtype, copy=False)
    reshape_shape = []
    for dim in cropped_shape:
        reshape_shape.extend([dim // factor, factor])
    axes = tuple(range(1, len(reshape_shape), 2))
    return arr.reshape(reshape_shape).mean(axis=axes, dtype=np.float32).astype(dtype, copy=False)


def save_downsampled_whole_roi_fields(cfg: dict, sample_key: str, result: dict) -> tuple[bool, str]:
    output_mode = str(cfg.get("whole_roi_field_output", "none")).lower()
    if output_mode not in {"downsampled", "downsample", "2x"}:
        return False, ""

    potential = result.get("potential")
    flux_mag = result.get("flux_magnitude")
    if potential is None or flux_mag is None:
        return False, ""

    factor = int(cfg.get("whole_roi_downsample_factor", 2))
    dtype = str(cfg.get("field_dtype", "float32"))
    out_dir = sample_fields_dir(cfg, sample_key) / "whole_roi"
    out_dir.mkdir(parents=True, exist_ok=True)

    potential_ds = downsample_mean(np.asarray(potential), factor, dtype)
    flux_mag_ds = downsample_mean(np.asarray(flux_mag), factor, dtype)
    np.savez_compressed(
        out_dir / "whole_roi_downsampled_fields.npz",
        potential=potential_ds,
        concentration=potential_ds,
        flux_magnitude=flux_mag_ds,
    )
    save_scalar_orthogonal_slices(
        potential_ds,
        out_dir,
        "potential_downsampled",
        f"{sample_key} whole ROI potential downsampled {factor}x",
    )
    save_scalar_orthogonal_slices(
        flux_mag_ds,
        out_dir,
        "flux_magnitude_downsampled",
        f"{sample_key} whole ROI flux magnitude downsampled {factor}x",
        cmap="magma",
    )

    metadata = {
        "sample": sample_key,
        "source": "whole_roi",
        "axis_order": cfg.get("puma_axis_order", "x_y_z"),
        "downsample_factor": factor,
        "dtype": dtype,
        "arrays": ["potential", "concentration", "flux_magnitude"],
        "field_file": str(out_dir / "whole_roi_downsampled_fields.npz"),
        "potential_shape": list(potential_ds.shape),
        "flux_magnitude_shape": list(flux_mag_ds.shape),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return True, str(out_dir)


def run_one_sample(cfg: dict, sample_key: str, force: bool = False) -> dict:
    out_dir = sample_results_dir(cfg, sample_key)
    result_path = out_dir / "whole_roi_transport.csv"
    if not force and result_is_complete(result_path):
        print(f"[{sample_key}] found complete {result_path}; skipping whole-ROI recomputation.", flush=True)
        return pd.read_csv(result_path).iloc[0].to_dict()

    label_path = sample_label_path(cfg, sample_key)
    if not label_path.exists():
        raise FileNotFoundError(f"Missing {label_path}; run 01_validate_and_merge_masks.py first.")

    print(f"[{sample_key}] reading {label_path}", flush=True)
    label = read_volume_as_zyx(label_path, cfg.get("tiff_axis_order", "z_y_x"))
    z, y, x = label.shape
    fractions = label_fractions(label, cfg["labels"])
    row = {
        "sample": sample_key,
        "calculation": "whole_roi",
        "x0": 0,
        "x1": int(x),
        "y0": 0,
        "y1": int(y),
        "z0": 0,
        "z1": int(z),
        "shape_x": int(x),
        "shape_y": int(y),
        "shape_z": int(z),
        "voxel_count": int(label.size),
        "voxel_size_um": cfg["voxel_size_um"],
        "physical_size_x_um": float(x) * float(cfg["voxel_size_um"]),
        "physical_size_y_um": float(y) * float(cfg["voxel_size_um"]),
        "physical_size_z_um": float(z) * float(cfg["voxel_size_um"]),
        **fractions,
        "direction": cfg["through_plane_axis"],
        "side_bc": cfg.get("side_bc", "symmetric"),
        "field_saved": False,
    }

    start = time.perf_counter()
    mem_before = current_memory_mb()
    try:
        print(
            f"[{sample_key}] whole ROI shape zyx={label.shape}, direction={cfg['through_plane_axis']}; "
            f"starting PuMA solve with whole_roi_field_output={cfg.get('whole_roi_field_output', 'none')}",
            flush=True,
        )
        result = compute_transport(label, cfg, prefer_electrical=True)
        field_saved, field_path = save_downsampled_whole_roi_fields(cfg, sample_key, result)
        row.update(
            {
                "keff_norm": result["keff_norm"],
                "solver_type": result["solver_name"],
                "runtime_s": result.get("runtime_s", time.perf_counter() - start),
                "memory_mb": current_memory_mb(),
                "field_saved": field_saved,
                "field_path": field_path,
                "whole_roi_field_output": cfg.get("whole_roi_field_output", "none"),
                "whole_roi_downsample_factor": cfg.get("whole_roi_downsample_factor", ""),
                "status": "ok",
                "error": "",
            }
        )
    except Exception as exc:
        row.update(
            {
                "keff_norm": math.nan,
                "solver_type": "",
                "runtime_s": time.perf_counter() - start,
                "memory_mb": mem_before,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_rows_csv(result_path, [row])
        raise
    finally:
        del label
        gc.collect()

    write_rows_csv(result_path, [row])
    print(f"[{sample_key}] saved {result_path}", flush=True)
    print(f"[{sample_key}] keff_norm={row['keff_norm']:.8g}, runtime_s={row['runtime_s']:.1f}", flush=True)
    return row


def write_available_whole_roi_summary(cfg: dict) -> None:
    rows = []
    for sample_key in cfg["samples"]:
        path = sample_results_dir(cfg, sample_key) / "whole_roi_transport.csv"
        if path.exists():
            rows.append(pd.read_csv(path).iloc[0].to_dict())
    if rows:
        write_rows_csv(Path(cfg["_root"]) / cfg.get("results_dir", "results") / "whole_roi_transport.csv", rows)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    sample_keys = [args.sample] if args.sample else list(cfg["samples"].keys())
    for sample_key in sample_keys:
        run_one_sample(cfg, sample_key, force=args.force)
    write_available_whole_roi_summary(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

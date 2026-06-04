from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import (
    available_pumapy_functions,
    compute_transport,
    current_memory_mb,
    load_config,
    read_volume_as_zyx,
    sample_label_path,
    sample_fields_dir,
    sample_results_dir,
    save_flux_slice_png,
    save_npz,
    save_scalar_orthogonal_slices,
    save_scalar_slice_png,
    write_rows_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PuMA transport calculation for all regular sub-volumes.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--sample", choices=["WM", "PFDT"], help="Run only one sample.")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--workers", type=int, help="Number of parallel worker processes. Default: config workers or 1.")
    return parser.parse_args()


def crop_from_row(label: np.ndarray, row: pd.Series) -> np.ndarray:
    return label[int(row.z0): int(row.z1), int(row.y0): int(row.y1), int(row.x0): int(row.x1)]


def cast_field(field, dtype: str):
    if field is None:
        return None
    if isinstance(field, (list, tuple)):
        return np.asarray(field, dtype=dtype)
    return np.asarray(field, dtype=dtype)


def save_subvolume_fields(cfg: dict, sample_key: str, subvolume_id: str, result: dict) -> None:
    mode = str(cfg.get("field_output", "representative")).lower()
    if mode != "all":
        return
    dtype = str(cfg.get("field_dtype", "float32"))
    direction = cfg["through_plane_axis"]
    out_dir = sample_fields_dir(cfg, sample_key) / subvolume_id
    potential = cast_field(result.get("potential"), dtype)
    flux = cast_field(result.get("flux"), dtype)
    flux_mag = cast_field(result.get("flux_magnitude"), dtype)
    save_npz(
        out_dir / "transport_fields.npz",
        concentration=potential,
        potential=potential,
        flux=flux,
        flux_magnitude=flux_mag,
    )
    save_scalar_slice_png(potential, out_dir / "concentration_center_slice.png", f"{subvolume_id} concentration", direction)
    save_flux_slice_png(flux_mag, out_dir / "flux_magnitude_center_slice.png", f"{subvolume_id} flux magnitude", direction)
    save_scalar_orthogonal_slices(potential, out_dir, "concentration", f"{subvolume_id} concentration")
    save_scalar_orthogonal_slices(flux_mag, out_dir, "flux_magnitude", f"{subvolume_id} flux magnitude", cmap="magma")


def run_subvolume_task(cfg: dict, sample_key: str, label_path: str, row_dict: dict, task_index: int, total: int) -> dict:
    label = read_volume_as_zyx(Path(label_path), "z_y_x")
    row = pd.Series(row_dict)
    block = crop_from_row(label, row)
    start = time.perf_counter()
    mem_before = current_memory_mb()
    result_row = dict(row_dict)
    try:
        result = compute_transport(block, cfg, prefer_electrical=True)
        save_subvolume_fields(cfg, sample_key, str(row.subvolume_id), result)
        result_row.update(
            {
                "keff_norm": result["keff_norm"],
                "solver_type": result["solver_name"],
                "direction": cfg["through_plane_axis"],
                "side_bc": cfg.get("side_bc", "symmetric"),
                "runtime_s": result.get("runtime_s", time.perf_counter() - start),
                "memory_mb": current_memory_mb(),
                "status": "ok",
                "error": "",
                "task_index": task_index,
            }
        )
    except Exception as exc:
        result_row.update(
            {
                "keff_norm": math.nan,
                "solver_type": "",
                "direction": cfg["through_plane_axis"],
                "side_bc": cfg.get("side_bc", "symmetric"),
                "runtime_s": time.perf_counter() - start,
                "memory_mb": mem_before,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "task_index": task_index,
            }
        )
    return result_row


def completed_results_are_usable(results_df: pd.DataFrame, subvolumes: pd.DataFrame) -> bool:
    if results_df.empty or len(results_df) != len(subvolumes):
        return False
    if "subvolume_id" not in results_df or "keff_norm" not in results_df:
        return False
    expected = set(subvolumes["subvolume_id"].astype(str))
    actual = set(results_df["subvolume_id"].astype(str))
    if actual != expected:
        return False
    if "status" in results_df and not (results_df["status"].astype(str) == "ok").all():
        return False
    return bool(np.isfinite(results_df["keff_norm"].astype(float)).all())


def save_representative_selection(cfg: dict, sample_key: str, out_dir: Path, results_df: pd.DataFrame, label_path: Path) -> None:
    ok = results_df[np.isfinite(results_df["keff_norm"].astype(float))]
    if ok.empty:
        print(f"[{sample_key}] no successful PuMA results; skipping representative flux map.")
        return

    median = float(ok["keff_norm"].median())
    representative_pos = int(np.argmin(np.abs(ok["keff_norm"].to_numpy(float) - median)))
    representative = ok.iloc[representative_pos]

    rep_dir = out_dir / "representative_flux"
    rep_dir.mkdir(parents=True, exist_ok=True)
    representative_field = rep_dir / "median_representative_fields.npz"
    metadata = {
        "sample": sample_key,
        "subvolume_id": representative["subvolume_id"],
        "keff_norm": float(representative["keff_norm"]),
        "group_median_keff_norm": median,
        "coordinates": {
            "x0": int(representative["x0"]),
            "x1": int(representative["x1"]),
            "y0": int(representative["y0"]),
            "y1": int(representative["y1"]),
            "z0": int(representative["z0"]),
            "z1": int(representative["z1"]),
        },
        "direction": cfg["through_plane_axis"],
    }

    existing_field = sample_fields_dir(cfg, sample_key) / str(representative["subvolume_id"]) / "transport_fields.npz"
    if existing_field.exists():
        metadata["solver_type"] = str(representative.get("solver_type", "electrical"))
        metadata["source"] = str(existing_field)
        metadata["fields_saved"] = True
        (rep_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        shutil.copy2(existing_field, rep_dir / "median_representative_fields.npz")
        for name in [
            "concentration_center_slice.png",
            "flux_magnitude_center_slice.png",
            "concentration_xy_center_z.png",
            "concentration_xz_center_y.png",
            "concentration_yz_center_x.png",
            "flux_magnitude_xy_center_z.png",
            "flux_magnitude_xz_center_y.png",
            "flux_magnitude_yz_center_x.png",
        ]:
            src = existing_field.parent / name
            if src.exists():
                shutil.copy2(src, rep_dir / name)
        print(
            f"[{sample_key}] representative sub-volume: {representative['subvolume_id']} "
            f"(median={median:.8g}); reused saved field",
            flush=True,
        )
        print(f"[{sample_key}] saved representative fields in {rep_dir}", flush=True)
        return

    if representative_field.exists() and str(cfg.get("representative_output", "auto")).lower() != "force_recompute":
        metadata["solver_type"] = str(representative.get("solver_type", "electrical"))
        metadata["source"] = str(representative_field)
        metadata["fields_saved"] = True
        metadata["note"] = "Existing representative field was reused. Set representative_output='force_recompute' to overwrite it."
        (rep_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(
            f"[{sample_key}] representative sub-volume: {representative['subvolume_id']} "
            f"(median={median:.8g}); existing representative field reused",
            flush=True,
        )
        return

    representative_output = str(cfg.get("representative_output", "auto")).lower()
    if representative_output not in {"recompute", "compute", "solve", "force_recompute"}:
        metadata["solver_type"] = str(representative.get("solver_type", "electrical"))
        metadata["source"] = "transport_results.csv"
        metadata["fields_saved"] = False
        metadata["note"] = (
            "Representative was selected from existing keff_norm values. Field arrays were not recomputed; "
            "set representative_output='recompute' only when a new representative field map is required."
        )
        (rep_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(
            f"[{sample_key}] representative sub-volume: {representative['subvolume_id']} "
            f"(median={median:.8g}); metadata only, no field recomputation",
            flush=True,
        )
        return

    label = read_volume_as_zyx(label_path, "z_y_x")
    block = crop_from_row(label, representative)
    result = compute_transport(block, cfg, prefer_electrical=True)
    metadata["solver_type"] = result["solver_name"]
    metadata["fields_saved"] = True
    metadata["source"] = "representative_recompute"
    (rep_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    dtype = str(cfg.get("field_dtype", "float32"))
    save_npz(
        representative_field,
        potential=cast_field(result.get("potential"), dtype),
        flux=cast_field(result.get("flux"), dtype),
        flux_magnitude=cast_field(result.get("flux_magnitude"), dtype),
    )
    save_flux_slice_png(
        result.get("flux_magnitude"),
        rep_dir / "flux_magnitude_center_slice.png",
        f"{sample_key} median representative flux magnitude",
        cfg["through_plane_axis"],
    )
    print(f"[{sample_key}] representative sub-volume: {representative['subvolume_id']} (median={median:.8g})")
    print(f"[{sample_key}] saved representative fields in {rep_dir}")


def run_one_sample(cfg: dict, sample_key: str) -> None:
    out_dir = sample_results_dir(cfg, sample_key)
    label_path = sample_label_path(cfg, sample_key)
    subvol_path = out_dir / "subvolumes.csv"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing {label_path}; run 01_validate_and_merge_masks.py first.")
    if not subvol_path.exists():
        raise FileNotFoundError(f"Missing {subvol_path}; run 03_generate_subvolumes.py first.")

    subvolumes = pd.read_csv(subvol_path)
    results_path = out_dir / "transport_results.csv"
    if bool(cfg.get("skip_completed", True)) and results_path.exists():
        existing_results = pd.read_csv(results_path)
        if completed_results_are_usable(existing_results, subvolumes):
            print(f"\n[{sample_key}] found complete {results_path}; skipping PuMA recomputation.", flush=True)
            save_representative_selection(cfg, sample_key, out_dir, existing_results, label_path)
            return

    rows = []
    workers = int(cfg.get("workers", 1))
    print(
        f"\n[{sample_key}] running {len(subvolumes)} sub-volumes with direction={cfg['through_plane_axis']}, "
        f"workers={workers}",
        flush=True,
    )

    if workers <= 1:
        for i, row in subvolumes.iterrows():
            result_row = run_subvolume_task(cfg, sample_key, str(label_path), row.to_dict(), i, len(subvolumes))
            rows.append(result_row)
            if result_row["status"] == "ok":
                print(f"  {i + 1:4d}/{len(subvolumes)} {row.subvolume_id}: keff_norm={result_row['keff_norm']:.8g}", flush=True)
            else:
                print(f"  {i + 1:4d}/{len(subvolumes)} {row.subvolume_id}: ERROR {result_row['error']}", flush=True)
                print("  PuMA conductivity/workspace functions:", ", ".join(available_pumapy_functions()), flush=True)
                if not cfg.get("continue_on_error", True):
                    raise RuntimeError(result_row["error"])
    else:
        futures = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for i, row in subvolumes.iterrows():
                futures.append(
                    executor.submit(run_subvolume_task, cfg, sample_key, str(label_path), row.to_dict(), i, len(subvolumes))
                )
            completed = 0
            for future in as_completed(futures):
                result_row = future.result()
                rows.append(result_row)
                completed += 1
                subvolume_id = result_row["subvolume_id"]
                if result_row["status"] == "ok":
                    print(
                        f"  {completed:4d}/{len(subvolumes)} {subvolume_id}: "
                        f"keff_norm={result_row['keff_norm']:.8g}, runtime_s={result_row['runtime_s']:.1f}",
                        flush=True,
                    )
                else:
                    print(f"  {completed:4d}/{len(subvolumes)} {subvolume_id}: ERROR {result_row['error']}", flush=True)
                    if not cfg.get("continue_on_error", True):
                        raise RuntimeError(result_row["error"])

    rows = sorted(rows, key=lambda row: int(row.get("task_index", 0)))

    fieldnames = list(rows[0].keys()) if rows else []
    write_rows_csv(results_path, rows, fieldnames)
    print(f"[{sample_key}] saved {results_path}")

    results_df = pd.DataFrame(rows)
    save_representative_selection(cfg, sample_key, out_dir, results_df, label_path)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["continue_on_error"] = args.continue_on_error
    cfg["workers"] = args.workers if args.workers is not None else int(cfg.get("workers", 1))
    try:
        import pumapy  # noqa: F401
    except Exception as exc:
        print(f"Could not import pumapy: {exc}")
        return 2

    sample_keys = [args.sample] if args.sample else list(cfg["samples"].keys())
    for sample_key in sample_keys:
        run_one_sample(cfg, sample_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

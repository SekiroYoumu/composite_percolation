from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
SAMPLES = ("WM", "PFDT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute all 30 um in-plane sub-volume fields and summarize SE-only spatial CV."
    )
    parser.add_argument("--project-root", default="/mnt/percolation_disk/percolation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--source-results", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--samples", nargs="+", choices=SAMPLES, default=list(SAMPLES))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads-per-worker", type=int, default=8)
    parser.add_argument("--direction", choices=["x", "y", "z"], default="y")
    parser.add_argument("--se-label", type=int, default=2)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Reuse completed field and metrics files (default).",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Recompute every selected subvolume even when outputs already exist.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument("--no-save-flux-vector", action="store_true")
    parser.add_argument("--max-subvolumes", type=int, default=None)
    return parser.parse_args()


def configure_threads(n: int) -> None:
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = str(n)


def import_pipeline(project_root: Path):
    scripts_dir = project_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import pipeline_common as pc  # noqa: PLC0415

    return pc


def crop_zyx(label_zyx: np.ndarray, row: pd.Series) -> np.ndarray:
    return label_zyx[
        int(row.z0): int(row.z1),
        int(row.y0): int(row.y1),
        int(row.x0): int(row.x1),
    ]


def to_xyz(label_zyx: np.ndarray) -> np.ndarray:
    return np.transpose(label_zyx, (2, 1, 0))


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return values[np.isfinite(values)]


def metric_stats(values: np.ndarray) -> dict[str, float]:
    values = finite_values(values).astype(np.float64, copy=False)
    if values.size == 0:
        return {"mean": math.nan, "std": math.nan, "cv": math.nan, "median": math.nan, "p10": math.nan, "p90": math.nan}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    cv = float(std / mean) if mean != 0 else math.nan
    return {
        "mean": mean,
        "std": std,
        "cv": cv,
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def maybe_load_done(metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_fields(
    field_path: Path,
    potential: np.ndarray,
    flux: np.ndarray | None,
    flux_magnitude: np.ndarray,
    save_flux_vector: bool,
) -> None:
    field_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "potential": potential.astype(np.float32, copy=False),
        "concentration": potential.astype(np.float32, copy=False),
        "flux_magnitude": flux_magnitude.astype(np.float32, copy=False),
    }
    if save_flux_vector and flux is not None:
        flux = np.asarray(flux)
        if flux.ndim == 4 and flux.shape[-1] == 3:
            arrays["flux"] = flux.astype(np.float32, copy=False)
            arrays["flux_x"] = arrays["flux"][..., 0]
            arrays["flux_y"] = arrays["flux"][..., 1]
            arrays["flux_z"] = arrays["flux"][..., 2]
        else:
            arrays["flux"] = flux.astype(np.float32, copy=False)
    np.savez(field_path, **arrays)


def task(payload: dict) -> dict:
    configure_threads(int(payload["threads_per_worker"]))
    project_root = Path(payload["project_root"])
    pc = import_pipeline(project_root)

    cfg = pc.load_config(payload["config"])
    cfg["through_plane_axis"] = payload["direction"]

    sample = payload["sample"]
    row = pd.Series(payload["row"])
    subvolume_id = str(row.subvolume_id)
    out_dir = Path(payload["output_root"]) / sample / subvolume_id
    field_path = out_dir / "fields_float32.npz"
    metrics_path = out_dir / "metrics.json"

    if payload["resume"] and field_path.exists() and metrics_path.exists():
        done = maybe_load_done(metrics_path)
        if done and done.get("status") == "ok":
            done["resumed"] = True
            return done

    start = time.perf_counter()
    label_zyx = tiff.imread(payload["label_path"])
    block_zyx = crop_zyx(label_zyx, row)
    label_xyz = to_xyz(block_zyx)
    se_mask = label_xyz == int(payload["se_label"])
    del label_zyx

    result = pc.compute_transport(block_zyx, cfg, prefer_electrical=True)
    potential = np.asarray(result["potential"], dtype=np.float32)
    flux = result.get("flux")
    flux_arr = None if flux is None else np.asarray(flux, dtype=np.float32)
    flux_magnitude = np.asarray(result["flux_magnitude"], dtype=np.float32)

    common_shape = tuple(min(potential.shape[i], flux_magnitude.shape[i], label_xyz.shape[i]) for i in range(3))
    slicer = tuple(slice(0, n) for n in common_shape)
    potential = potential[slicer]
    flux_magnitude = flux_magnitude[slicer]
    label_xyz = label_xyz[slicer]
    se_mask = label_xyz == int(payload["se_label"])

    if flux_arr is not None and flux_arr.ndim == 4:
        flux_arr = flux_arr[slicer + (slice(None),)]
        abs_j = np.abs(flux_arr[..., AXIS_INDEX[payload["direction"]]]).astype(np.float32, copy=False)
    elif flux_arr is not None:
        flux_arr = flux_arr[slicer]
        abs_j = np.abs(flux_arr).astype(np.float32, copy=False)
    else:
        abs_j = None

    se_count = int(np.count_nonzero(se_mask))
    row_out: dict[str, float | int | str | bool] = {
        "sample": sample,
        "subvolume_id": subvolume_id,
        "status": "ok",
        "direction": payload["direction"],
        "solver_type": str(result.get("solver_name", "")),
        "keff_norm": float(result["keff_norm"]),
        "runtime_s": float(time.perf_counter() - start),
        "n_se_voxels": se_count,
        "x0": int(row.x0),
        "x1": int(row.x1),
        "y0": int(row.y0),
        "y1": int(row.y1),
        "z0": int(row.z0),
        "z1": int(row.z1),
        "resumed": False,
    }

    for prefix, scalar in [
        ("concentration", potential),
        ("flux_magnitude", flux_magnitude),
        (f"abs_J{payload['direction']}", abs_j),
    ]:
        if scalar is None:
            continue
        values = scalar[se_mask]
        if prefix != "concentration":
            values = np.abs(values)
        stats = metric_stats(values)
        for key, value in stats.items():
            row_out[f"{prefix}_{key}"] = value

    save_fields(
        field_path,
        potential=potential,
        flux=flux_arr,
        flux_magnitude=flux_magnitude,
        save_flux_vector=not bool(payload["no_save_flux_vector"]),
    )
    row_out["field_path"] = str(field_path)
    metrics_path.write_text(json.dumps(row_out, indent=2), encoding="utf-8")

    del block_zyx, label_xyz, se_mask, potential, flux_arr, flux_magnitude, result
    gc.collect()
    return row_out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def summarize(rows: list[dict], output_root: Path) -> None:
    df = pd.DataFrame(rows)
    ok = df[df["status"].astype(str) == "ok"].copy()
    if ok.empty:
        return
    metrics = [
        "concentration_cv",
        "flux_magnitude_cv",
        "abs_Jy_cv",
        "keff_norm",
    ]
    summary_rows = []
    for sample, group in ok.groupby("sample"):
        for metric in metrics:
            if metric not in group:
                continue
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            if vals.empty:
                continue
            summary_rows.append(
                {
                    "sample": sample,
                    "metric": metric,
                    "n": int(vals.size),
                    "mean": float(vals.mean()),
                    "sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                    "median": float(vals.median()),
                    "cv_across_subvolumes": float(vals.std(ddof=1) / vals.mean()) if vals.size > 1 and vals.mean() else math.nan,
                }
            )
    pd.DataFrame(summary_rows).to_csv(output_root / "field_cv_summary.csv", index=False)


def scatter_panel(ax, df: pd.DataFrame, metric: str, ylabel: str, colors: dict[str, str]) -> None:
    rng = np.random.default_rng(20260617)
    samples = list(SAMPLES)
    for i, sample in enumerate(samples):
        vals = pd.to_numeric(df.loc[df["sample"] == sample, metric], errors="coerce").dropna().to_numpy(float)
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.06, 0.06, size=vals.size)
        ax.scatter(np.full(vals.size, i) + jitter, vals, s=54, color=colors[sample], edgecolor="black", linewidth=0.6, zorder=3)
        mean = float(vals.mean())
        ax.hlines(mean, i - 0.22, i + 0.22, color="black", linewidth=2.0, zorder=4)
    ax.set_xticks(range(len(samples)), samples)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)


def plot_cv(output_root: Path) -> None:
    csv_path = output_root / "field_cv_results.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df = df[df["status"].astype(str) == "ok"].copy()
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"WM": "#4C78A8", "PFDT": "#F58518"}

    panels = [
        ("keff_norm", "Normalized effective transport"),
        ("concentration_cv", "Concentration spatial CV"),
        ("flux_magnitude_cv", "Flux magnitude spatial CV"),
        ("abs_Jy_cv", "|Jy| spatial CV"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(12.5, 3.6), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, panels):
        scatter_panel(ax, df, metric, ylabel, colors)
    fig.savefig(fig_dir / "keff_concentration_flux_cv_4dot_mean.png", dpi=300)
    plt.close(fig)

    for metric, ylabel in panels:
        fig, ax = plt.subplots(figsize=(3.4, 3.6), constrained_layout=True)
        scatter_panel(ax, df, metric, ylabel, colors)
        fig.savefig(fig_dir / f"{metric}_4dot_mean.png", dpi=300)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    configure_threads(args.threads_per_worker)

    project_root = Path(args.project_root).resolve()
    config_path = Path(args.config) if args.config else project_root / "config.json"
    source_results = Path(args.source_results) if args.source_results else project_root / "results"
    output_root = Path(args.output_root) if args.output_root else project_root / "runs" / "30um-in-plane-all-fields-cv"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"project_root={project_root}", flush=True)
    print(f"source_results={source_results}", flush=True)
    print(f"output_root={output_root}", flush=True)
    print(f"workers={args.workers}, threads_per_worker={args.threads_per_worker}, direction={args.direction}", flush=True)
    shutil.copy2(config_path, output_root / "source_config.json")

    payloads = []
    for sample in args.samples:
        label_path = source_results / sample / "label_zyx.tif"
        subvol_path = source_results / sample / "subvolumes.csv"
        transport_path = source_results / sample / "transport_results.csv"
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        if not subvol_path.exists():
            raise FileNotFoundError(subvol_path)
        subvols = pd.read_csv(subvol_path)
        if transport_path.exists():
            transport = pd.read_csv(transport_path)[["subvolume_id", "keff_norm"]]
            subvols = subvols.drop(columns=["keff_norm"], errors="ignore").merge(transport, on="subvolume_id", how="left")
        if args.max_subvolumes is not None:
            subvols = subvols.head(args.max_subvolumes)
        for _, row in subvols.iterrows():
            payloads.append(
                {
                    "project_root": str(project_root),
                    "config": str(config_path),
                    "source_results": str(source_results),
                    "output_root": str(output_root),
                    "sample": sample,
                    "label_path": str(label_path),
                    "row": row.to_dict(),
                    "direction": args.direction,
                    "se_label": args.se_label,
                    "resume": args.resume,
                    "threads_per_worker": args.threads_per_worker,
                    "no_save_flux_vector": args.no_save_flux_vector,
                }
            )

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(task, payload) for payload in payloads]
        for i, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            if row.get("status") == "ok":
                print(
                    f"{i:02d}/{len(futures)} {row['sample']} {row['subvolume_id']} "
                    f"keff={float(row['keff_norm']):.8g} "
                    f"concCV={float(row.get('concentration_cv', math.nan)):.4f} "
                    f"fluxCV={float(row.get('flux_magnitude_cv', math.nan)):.4f} "
                    f"runtime={float(row.get('runtime_s', 0.0)):.1f}s "
                    f"{'resumed' if row.get('resumed') else 'done'}",
                    flush=True,
                )
            else:
                print(f"{i:02d}/{len(futures)} ERROR {row}", flush=True)
            write_csv(output_root / "field_cv_results.partial.csv", sorted(rows, key=lambda r: (str(r.get("sample")), str(r.get("subvolume_id")))))

    rows = sorted(rows, key=lambda r: (str(r.get("sample")), str(r.get("subvolume_id"))))
    write_csv(output_root / "field_cv_results.csv", rows)
    summarize(rows, output_root)
    plot_cv(output_root)
    print(f"Saved {output_root / 'field_cv_results.csv'}", flush=True)
    print(f"Saved figures in {output_root / 'figures'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

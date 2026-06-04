from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import center_slice, figures_dir, load_config, sample_results_dir, write_rows_csv


PHASES = ["CAM", "SE-rich", "Void/carbon-rich"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot transport and phase-fraction summary figures.")
    parser.add_argument("--config", default="config.json")
    return parser.parse_args()


def read_results(cfg: dict) -> pd.DataFrame:
    frames = []
    for sample_key in cfg["samples"]:
        path = sample_results_dir(cfg, sample_key) / "transport_results.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run 04_run_puma_batch.py first.")
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["status"].fillna("ok") == "ok"].copy()
    df["keff_norm"] = pd.to_numeric(df["keff_norm"], errors="coerce")
    return df[np.isfinite(df["keff_norm"])]


def scatter_with_mean_sd(df: pd.DataFrame, out: Path) -> None:
    samples = list(df["sample"].drop_duplicates())
    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    rng = np.random.default_rng(123)
    for i, sample in enumerate(samples, start=1):
        values = df.loc[df["sample"] == sample, "keff_norm"].to_numpy(float)
        x = np.full(values.size, i, dtype=float) + rng.uniform(-0.08, 0.08, size=values.size)
        ax.scatter(x, values, s=42, alpha=0.85, label=sample)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        ax.errorbar(i, mean, yerr=sd, fmt="o", ms=8, color="black", capsize=6, lw=1.5)
    ax.set_xticks(range(1, len(samples) + 1), samples)
    ax.set_ylabel("Normalized effective ionic transport coefficient")
    ax.set_xlabel("Sample")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def phase_fraction_scatter(df: pd.DataFrame, out: Path) -> None:
    samples = list(df["sample"].drop_duplicates())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True, sharey=True)
    rng = np.random.default_rng(456)
    for ax, phase in zip(axes, PHASES):
        col = f"{phase} fraction"
        for i, sample in enumerate(samples, start=1):
            values = df.loc[df["sample"] == sample, col].to_numpy(float)
            x = np.full(values.size, i, dtype=float) + rng.uniform(-0.08, 0.08, size=values.size)
            ax.scatter(x, values, s=36, alpha=0.85)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            ax.errorbar(i, mean, yerr=sd, fmt="o", ms=7, color="black", capsize=5, lw=1.3)
        ax.set_title(phase)
        ax.set_xticks(range(1, len(samples) + 1), samples)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Phase fraction")
    fig.savefig(out, dpi=220)
    plt.close(fig)


def representative_flux_maps(cfg: dict, out_linear: Path, out_log: Path) -> None:
    entries = []
    direction = cfg["through_plane_axis"].lower()
    for sample_key in cfg["samples"]:
        rep_dir = sample_results_dir(cfg, sample_key) / "representative_flux"
        npz_path = rep_dir / "median_representative_fields.npz"
        meta_path = rep_dir / "metadata.json"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        if "flux_magnitude" not in data:
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        entries.append((sample_key, np.asarray(data["flux_magnitude"], dtype=float), meta))
    if not entries:
        return

    linear_slices = [center_slice(arr, direction) for _, arr, _ in entries]
    finite = np.concatenate([img[np.isfinite(img)].ravel() for img in linear_slices if np.isfinite(img).any()])
    if finite.size == 0:
        return
    vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    eps = max(vmax * 1e-12, 1e-30)
    log_slices = [np.log10(img + eps) for img in linear_slices]
    log_finite = np.concatenate([img[np.isfinite(img)].ravel() for img in log_slices])
    log_vmin, log_vmax = float(np.nanmin(log_finite)), float(np.nanmax(log_finite))

    for out, images, lo, hi, label in [
        (out_linear, linear_slices, vmin, vmax, "Flux magnitude"),
        (out_log, log_slices, log_vmin, log_vmax, "log10 flux magnitude"),
    ]:
        fig, axes = plt.subplots(1, len(entries), figsize=(5 * len(entries), 4.5), constrained_layout=True)
        if len(entries) == 1:
            axes = [axes]
        last = None
        for ax, (sample, _, meta), image in zip(axes, entries, images):
            last = ax.imshow(image.T, origin="lower", cmap="magma", interpolation="nearest", vmin=lo, vmax=hi)
            subtitle = meta.get("subvolume_id", "median representative")
            ax.set_title(f"{sample}\n{subtitle}")
            ax.set_axis_off()
        fig.colorbar(last, ax=axes, shrink=0.85, label=label)
        fig.savefig(out, dpi=220)
        plt.close(fig)


def representative_concentration_maps(cfg: dict, out: Path) -> None:
    entries = []
    direction = cfg["through_plane_axis"].lower()
    for sample_key in cfg["samples"]:
        rep_dir = sample_results_dir(cfg, sample_key) / "representative_flux"
        npz_path = rep_dir / "median_representative_fields.npz"
        meta_path = rep_dir / "metadata.json"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        key = "concentration" if "concentration" in data else "potential"
        if key not in data:
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        entries.append((sample_key, np.asarray(data[key], dtype=float), meta))
    if not entries:
        return

    slices = [center_slice(arr, direction) for _, arr, _ in entries]
    finite = np.concatenate([img[np.isfinite(img)].ravel() for img in slices if np.isfinite(img).any()])
    if finite.size == 0:
        return
    vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))

    fig, axes = plt.subplots(1, len(entries), figsize=(5 * len(entries), 4.5), constrained_layout=True)
    if len(entries) == 1:
        axes = [axes]
    last = None
    for ax, (sample, _, meta), image in zip(axes, entries, slices):
        last = ax.imshow(image.T, origin="lower", cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
        subtitle = meta.get("subvolume_id", "median representative")
        ax.set_title(f"{sample}\n{subtitle}")
        ax.set_axis_off()
    fig.colorbar(last, ax=axes, shrink=0.85, label="Potential / concentration field")
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_summary(df: pd.DataFrame, out: Path) -> None:
    rows = []
    for sample, g in df.groupby("sample", sort=False):
        values = g["keff_norm"].to_numpy(float)
        row = {
            "sample": sample,
            "n": int(values.size),
            "keff_norm_mean": float(np.mean(values)),
            "keff_norm_SD": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "keff_norm_median": float(np.median(values)),
        }
        for phase in PHASES:
            col = f"{phase} fraction"
            row[f"{phase}_fraction_mean"] = float(g[col].mean())
            row[f"{phase}_fraction_SD"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    write_rows_csv(out, rows)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    fig_dir = figures_dir(cfg)
    df = read_results(cfg)
    scatter_with_mean_sd(df, fig_dir / "keff_norm_scatter.png")
    phase_fraction_scatter(df, fig_dir / "phase_fraction_scatter.png")
    representative_flux_maps(cfg, fig_dir / "flux_magnitude_linear.png", fig_dir / "flux_magnitude_log.png")
    representative_concentration_maps(cfg, fig_dir / "concentration_linear.png")
    write_summary(df, fig_dir / "summary.csv")
    print(f"Saved figures and summary in {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

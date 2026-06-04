from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pipeline_common import figures_dir, load_config, orthogonal_center_slices, sample_results_dir, save_scalar_orthogonal_slices


PLANES = ["xy_center_z", "xz_center_y", "yz_center_x"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot orthogonal slices from saved median-representative fields.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--log-flux", action="store_true", default=True)
    return parser.parse_args()


def read_field_slices(npz_path: Path) -> dict[str, dict[str, np.ndarray]]:
    data = np.load(npz_path)
    potential_key = "concentration" if "concentration" in data.files else "potential"
    if potential_key not in data.files:
        raise KeyError(f"{npz_path} has neither concentration nor potential. Keys: {data.files}")
    if "flux_magnitude" not in data.files:
        raise KeyError(f"{npz_path} has no flux_magnitude. Keys: {data.files}")

    potential = np.asarray(data[potential_key], dtype=np.float32)
    flux_mag = np.asarray(data["flux_magnitude"], dtype=np.float32)
    return {
        "potential": orthogonal_center_slices(potential),
        "flux_magnitude": orthogonal_center_slices(flux_mag),
    }


def save_combined_grid(entries: list[tuple[str, dict[str, np.ndarray], dict]], out: Path, cmap: str, label: str,
                       log_scale: bool = False) -> None:
    if not entries:
        return

    images = []
    for _, slices, _ in entries:
        for plane in PLANES:
            image = np.asarray(slices[plane], dtype=np.float32)
            if log_scale:
                positive = image[np.isfinite(image) & (image > 0)]
                eps = max(float(np.nanmax(positive)) * 1e-12, 1e-30) if positive.size else 1e-30
                image = np.log10(image + eps)
            images.append(image)
    finite = np.concatenate([img[np.isfinite(img)].ravel() for img in images if np.isfinite(img).any()])
    if finite.size == 0:
        return
    vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))

    fig, axes = plt.subplots(len(entries), len(PLANES), figsize=(4.5 * len(PLANES), 4.0 * len(entries)),
                             constrained_layout=True)
    if len(entries) == 1:
        axes = np.asarray([axes])
    last = None
    for row, (sample, slices, meta) in enumerate(entries):
        subtitle = meta.get("subvolume_id", "median representative")
        for col, plane in enumerate(PLANES):
            image = np.asarray(slices[plane], dtype=np.float32)
            if log_scale:
                positive = image[np.isfinite(image) & (image > 0)]
                eps = max(float(np.nanmax(positive)) * 1e-12, 1e-30) if positive.size else 1e-30
                image = np.log10(image + eps)
            ax = axes[row, col]
            last = ax.imshow(image.T, origin="lower", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
            ax.set_title(f"{sample} {subtitle}\n{plane}")
            ax.set_axis_off()
    fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.85, label=label)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    fig_dir = figures_dir(cfg)

    potential_entries = []
    flux_entries = []
    for sample_key in cfg["samples"]:
        rep_dir = sample_results_dir(cfg, sample_key) / "representative_flux"
        npz_path = rep_dir / "median_representative_fields.npz"
        meta_path = rep_dir / "metadata.json"
        if not npz_path.exists():
            print(f"[{sample_key}] missing {npz_path}; skipping.")
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        slices = read_field_slices(npz_path)

        data = np.load(npz_path)
        potential_key = "concentration" if "concentration" in data.files else "potential"
        potential = np.asarray(data[potential_key], dtype=np.float32)
        flux_mag = np.asarray(data["flux_magnitude"], dtype=np.float32)
        save_scalar_orthogonal_slices(
            potential,
            rep_dir,
            "potential",
            f"{sample_key} representative potential",
            cmap="viridis",
        )
        save_scalar_orthogonal_slices(
            flux_mag,
            rep_dir,
            "flux_magnitude",
            f"{sample_key} representative flux magnitude",
            cmap="magma",
        )
        potential_entries.append((sample_key, slices["potential"], meta))
        flux_entries.append((sample_key, slices["flux_magnitude"], meta))
        print(f"[{sample_key}] saved orthogonal slices in {rep_dir}")

    save_combined_grid(
        potential_entries,
        fig_dir / "representative_potential_orthogonal_slices.png",
        cmap="viridis",
        label="Potential / concentration field",
    )
    save_combined_grid(
        flux_entries,
        fig_dir / "representative_flux_magnitude_orthogonal_slices_linear.png",
        cmap="magma",
        label="Flux magnitude",
    )
    if args.log_flux:
        save_combined_grid(
            flux_entries,
            fig_dir / "representative_flux_magnitude_orthogonal_slices_log.png",
            cmap="magma",
            label="log10 flux magnitude",
            log_scale=True,
        )

    print(f"Saved combined orthogonal-slice figures in {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

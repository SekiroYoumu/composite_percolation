from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pipeline_common import (
    figures_dir,
    load_config,
    orthogonal_center_slices,
    sample_results_dir,
    save_scalar_orthogonal_slices,
    voxel_size_by_axis,
)


PLANES = ["xy_center_z", "xz_center_y", "yz_center_x"]
PLANE_AXES = {
    "xy_center_z": ("x", "y"),
    "xz_center_y": ("x", "z"),
    "yz_center_x": ("y", "z"),
}


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


def display_image(image: np.ndarray, log_scale: bool = False) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if not log_scale:
        return arr
    positive = arr[np.isfinite(arr) & (arr > 0)]
    eps = max(float(np.nanmax(positive)) * 1e-12, 1e-30) if positive.size else 1e-30
    return np.log10(arr + eps)


def global_limits(entries: list[tuple[str, dict[str, dict[str, np.ndarray]], dict]], field: str,
                  log_scale: bool = False) -> tuple[float, float]:
    images = []
    for _, fields, _ in entries:
        for plane in PLANES:
            images.append(display_image(fields[field][plane], log_scale))
    finite = np.concatenate([img[np.isfinite(img)].ravel() for img in images if np.isfinite(img).any()])
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.nanmin(finite)), float(np.nanmax(finite))


def plane_extent_um(entries: list[tuple[str, dict[str, dict[str, np.ndarray]], dict]], plane: str,
                    voxel_um: dict[str, float]) -> tuple[float, float, float, float]:
    axis_h, axis_v = PLANE_AXES[plane]
    width = 0.0
    height = 0.0
    for _, fields, _ in entries:
        image = np.asarray(fields["potential"][plane])
        width = max(width, image.shape[0] * voxel_um[axis_h])
        height = max(height, image.shape[1] * voxel_um[axis_v])
    return 0.0, width, 0.0, height


def image_extent_um(image: np.ndarray, plane: str, voxel_um: dict[str, float]) -> tuple[float, float, float, float]:
    axis_h, axis_v = PLANE_AXES[plane]
    return 0.0, image.shape[0] * voxel_um[axis_h], 0.0, image.shape[1] * voxel_um[axis_v]


def save_potential_flux_comparison(entries: list[tuple[str, dict[str, dict[str, np.ndarray]], dict]], cfg: dict,
                                   out: Path, log_flux: bool = False,
                                   plane: str | None = None) -> None:
    if not entries:
        return

    voxel_um = voxel_size_by_axis(cfg)
    planes = [plane] if plane else PLANES
    potential_limits = global_limits(entries, "potential")
    flux_limits = global_limits(entries, "flux_magnitude", log_flux)
    nrows = 2 * len(planes)
    ncols = len(entries)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), squeeze=False,
                             constrained_layout=True)

    for plane_index, plane_name in enumerate(planes):
        axis_h, axis_v = PLANE_AXES[plane_name]
        common_extent = plane_extent_um(entries, plane_name, voxel_um)
        for col, (sample, fields, meta) in enumerate(entries):
            subtitle = meta.get("subvolume_id", "median representative")
            for row_offset, field, cmap, label, limits, use_log in [
                (0, "potential", "viridis", "Potential / concentration", potential_limits, False),
                (1, "flux_magnitude", "magma", "log10 flux magnitude" if log_flux else "Flux magnitude", flux_limits, log_flux),
            ]:
                row = 2 * plane_index + row_offset
                ax = axes[row, col]
                image_raw = np.asarray(fields[field][plane_name], dtype=np.float32)
                image = display_image(image_raw, use_log)
                im = ax.imshow(
                    image.T,
                    origin="lower",
                    cmap=cmap,
                    interpolation="nearest",
                    vmin=limits[0],
                    vmax=limits[1],
                    extent=image_extent_um(image_raw, plane_name, voxel_um),
                    aspect="equal",
                )
                ax.set_xlim(common_extent[0], common_extent[1])
                ax.set_ylim(common_extent[2], common_extent[3])
                ax.set_xlabel(f"{axis_h} (um)")
                ax.set_ylabel(f"{axis_v} (um)")
                ax.set_title(f"{sample} {subtitle}\n{plane_name} {label}")
                if col == ncols - 1:
                    fig.colorbar(im, ax=ax, shrink=0.82, label=label)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    fig_dir = figures_dir(cfg)

    potential_entries = []
    flux_entries = []
    comparison_entries = []
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
        comparison_entries.append((sample_key, slices, meta))
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

    save_potential_flux_comparison(
        comparison_entries,
        cfg,
        fig_dir / "representative_potential_flux_comparison.png",
        log_flux=args.log_flux,
    )
    for plane in PLANES:
        save_potential_flux_comparison(
            comparison_entries,
            cfg,
            fig_dir / f"representative_potential_flux_comparison_{plane}.png",
            log_flux=args.log_flux,
            plane=plane,
        )

    print(f"Saved combined orthogonal-slice figures in {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

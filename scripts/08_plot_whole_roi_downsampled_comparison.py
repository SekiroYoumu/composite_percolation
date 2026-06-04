from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pipeline_common import figures_dir, load_config, orthogonal_center_slices, sample_fields_dir, voxel_size_by_axis


PLANES = ["xy_center_z", "xz_center_y", "yz_center_x"]
PLANE_AXES = {
    "xy_center_z": ("x", "y"),
    "xz_center_y": ("x", "z"),
    "yz_center_x": ("y", "z"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot whole-ROI downsampled potential/flux comparison figures.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--log-flux", action="store_true", default=True)
    return parser.parse_args()


def display_image(image: np.ndarray, log_scale: bool = False) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if not log_scale:
        return arr
    positive = arr[np.isfinite(arr) & (arr > 0)]
    eps = max(float(np.nanmax(positive)) * 1e-12, 1e-30) if positive.size else 1e-30
    return np.log10(arr + eps)


def load_whole_roi_fields(cfg: dict) -> list[tuple[str, dict[str, dict[str, np.ndarray]], dict]]:
    entries = []
    for sample_key in cfg["samples"]:
        field_dir = sample_fields_dir(cfg, sample_key) / "whole_roi"
        npz_path = field_dir / "whole_roi_downsampled_fields.npz"
        meta_path = field_dir / "metadata.json"
        if not npz_path.exists():
            print(f"[{sample_key}] missing {npz_path}; skipping.")
            continue

        data = np.load(npz_path)
        potential_key = "concentration" if "concentration" in data.files else "potential"
        if potential_key not in data.files:
            raise KeyError(f"{npz_path} has neither concentration nor potential. Keys: {data.files}")
        if "flux_magnitude" not in data.files:
            raise KeyError(f"{npz_path} has no flux_magnitude. Keys: {data.files}")

        potential = np.asarray(data[potential_key], dtype=np.float32)
        flux_mag = np.asarray(data["flux_magnitude"], dtype=np.float32)
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        entries.append(
            (
                sample_key,
                {
                    "potential": orthogonal_center_slices(potential),
                    "flux_magnitude": orthogonal_center_slices(flux_mag),
                },
                meta,
            )
        )
    return entries


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


def downsampled_voxel_um(cfg: dict, entries: list[tuple[str, dict[str, dict[str, np.ndarray]], dict]]) -> dict[str, float]:
    voxel = voxel_size_by_axis(cfg)
    factor = None
    for _, _, meta in entries:
        if "downsample_factor" in meta:
            factor = int(meta["downsample_factor"])
            break
    if factor is None:
        factor = int(cfg.get("whole_roi_downsample_factor", 1))
    return {axis: value * factor for axis, value in voxel.items()}


def image_extent_um(image: np.ndarray, plane: str, voxel_um: dict[str, float]) -> tuple[float, float, float, float]:
    axis_h, axis_v = PLANE_AXES[plane]
    return 0.0, image.shape[0] * voxel_um[axis_h], 0.0, image.shape[1] * voxel_um[axis_v]


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


def save_comparison(entries: list[tuple[str, dict[str, dict[str, np.ndarray]], dict]], cfg: dict, out: Path,
                    log_flux: bool = False, plane: str | None = None) -> None:
    if not entries:
        return

    voxel_um = downsampled_voxel_um(cfg, entries)
    planes = [plane] if plane else PLANES
    potential_limits = global_limits(entries, "potential")
    flux_limits = global_limits(entries, "flux_magnitude", log_flux)
    nrows = 2 * len(planes)
    ncols = len(entries)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.9 * nrows), squeeze=False,
                             constrained_layout=True)

    for plane_index, plane_name in enumerate(planes):
        axis_h, axis_v = PLANE_AXES[plane_name]
        common_extent = plane_extent_um(entries, plane_name, voxel_um)
        for col, (sample, fields, meta) in enumerate(entries):
            factor = meta.get("downsample_factor", cfg.get("whole_roi_downsample_factor", ""))
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
                ax.set_title(f"{sample} whole ROI {factor}x\n{plane_name} {label}")
                if col == ncols - 1:
                    fig.colorbar(im, ax=ax, shrink=0.82, label=label)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    fig_dir = figures_dir(cfg)
    entries = load_whole_roi_fields(cfg)

    save_comparison(entries, cfg, fig_dir / "whole_roi_potential_flux_comparison.png", log_flux=args.log_flux)
    for plane in PLANES:
        save_comparison(
            entries,
            cfg,
            fig_dir / f"whole_roi_potential_flux_comparison_{plane}.png",
            log_flux=args.log_flux,
            plane=plane,
        )

    print(f"Saved whole-ROI comparison figures in {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

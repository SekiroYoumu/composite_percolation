from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_common import (  # noqa: E402
    load_config,
    orthogonal_center_slices,
    read_volume_as_zyx,
    root_path,
    sample_fields_dir,
    sample_label_path,
    sample_results_dir,
    transpose_between_orders,
    voxel_size_by_axis,
)


PLANES = ["xy_center_z", "xz_center_y", "yz_center_x"]
PLANE_AXES = {
    "xy_center_z": ("x", "y"),
    "xz_center_y": ("x", "z"),
    "yz_center_x": ("y", "z"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot SE-only potential and flux-magnitude comparison figures. "
            "Other phases are masked transparent and color limits are computed from SE pixels only."
        )
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--source", choices=["representative", "whole_roi", "both"], default="both")
    parser.add_argument("--se-threshold", type=float, default=0.5, help="Minimum SE fraction to display after downsampling.")
    parser.add_argument("--percentile-low", type=float, default=1.0, help="Lower percentile for SE-only color limits.")
    parser.add_argument("--percentile-high", type=float, default=99.0, help="Upper percentile for SE-only color limits.")
    parser.add_argument("--output-dir", default="results/viz/se_only")
    return parser.parse_args()


def load_npz_fields(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    potential_key = "concentration" if "concentration" in data.files else "potential"
    if potential_key not in data.files:
        raise KeyError(f"{npz_path} has no concentration/potential array. Keys: {data.files}")
    if "flux_magnitude" not in data.files:
        raise KeyError(f"{npz_path} has no flux_magnitude array. Keys: {data.files}")
    return np.asarray(data[potential_key], dtype=np.float32), np.asarray(data["flux_magnitude"], dtype=np.float32)


def crop_label_from_metadata(label_zyx: np.ndarray, meta: dict) -> np.ndarray:
    coords = meta.get("coordinates")
    if not coords:
        return label_zyx
    return label_zyx[
        int(coords["z0"]): int(coords["z1"]),
        int(coords["y0"]): int(coords["y1"]),
        int(coords["x0"]): int(coords["x1"]),
    ]


def crop_to_shape(array: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    slices = tuple(slice(0, min(dim, target)) for dim, target in zip(array.shape, shape))
    cropped = array[slices]
    final_slices = tuple(slice(0, dim) for dim in cropped.shape)
    if cropped.shape == shape:
        return cropped
    common = tuple(min(dim, target) for dim, target in zip(cropped.shape, shape))
    return cropped[tuple(slice(0, dim) for dim in common)]


def se_fraction_for_field(label_puma: np.ndarray, se_label: int, field_shape: tuple[int, int, int],
                          downsample_factor: int) -> np.ndarray:
    se = (label_puma == se_label).astype(np.float32)
    if downsample_factor <= 1:
        common = tuple(min(a, b) for a, b in zip(se.shape, field_shape))
        return se[tuple(slice(0, dim) for dim in common)]

    target_shape = tuple(int(dim) for dim in field_shape)
    crop_shape = tuple(min(se.shape[i], target_shape[i] * downsample_factor) for i in range(3))
    crop_shape = tuple((dim // downsample_factor) * downsample_factor for dim in crop_shape)
    se = se[tuple(slice(0, dim) for dim in crop_shape)]
    reshape_shape: list[int] = []
    for dim in crop_shape:
        reshape_shape.extend([dim // downsample_factor, downsample_factor])
    axes = tuple(range(1, len(reshape_shape), 2))
    frac = se.reshape(reshape_shape).mean(axis=axes, dtype=np.float32)
    common = tuple(min(a, b) for a, b in zip(frac.shape, target_shape))
    return frac[tuple(slice(0, dim) for dim in common)]


def load_entries(cfg: dict, source: str) -> list[dict]:
    labels = cfg["labels"]
    se_label = int(labels["SE-rich"])
    entries = []
    for sample_key in cfg["samples"]:
        label_path = sample_label_path(cfg, sample_key)
        if not label_path.exists():
            print(f"[{sample_key}] missing label volume {label_path}; skipping.")
            continue

        if source == "representative":
            field_dir = sample_results_dir(cfg, sample_key) / "representative_flux"
            npz_path = field_dir / "median_representative_fields.npz"
            meta_path = field_dir / "metadata.json"
            downsample_factor = 1
        elif source == "whole_roi":
            field_dir = sample_fields_dir(cfg, sample_key) / "whole_roi"
            npz_path = field_dir / "whole_roi_downsampled_fields.npz"
            meta_path = field_dir / "metadata.json"
            downsample_factor = int(cfg.get("whole_roi_downsample_factor", 1))
        else:
            raise ValueError(source)

        if not npz_path.exists():
            print(f"[{sample_key}] missing field file {npz_path}; skipping.")
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        downsample_factor = int(meta.get("downsample_factor", downsample_factor))
        potential, flux_mag = load_npz_fields(npz_path)
        common_shape = tuple(min(a, b) for a, b in zip(potential.shape, flux_mag.shape))
        potential = potential[tuple(slice(0, dim) for dim in common_shape)]
        flux_mag = flux_mag[tuple(slice(0, dim) for dim in common_shape)]

        label_zyx = read_volume_as_zyx(label_path, "z_y_x")
        label_zyx = crop_label_from_metadata(label_zyx, meta)
        label_puma = transpose_between_orders(label_zyx, "z_y_x", cfg.get("puma_axis_order", "x_y_z"))
        se_fraction = se_fraction_for_field(label_puma, se_label, common_shape, downsample_factor)
        common_shape = tuple(min(common_shape[i], se_fraction.shape[i]) for i in range(3))
        slicer = tuple(slice(0, dim) for dim in common_shape)

        entries.append(
            {
                "sample": sample_key,
                "source": source,
                "meta": meta,
                "downsample_factor": downsample_factor,
                "fields": {
                    "potential": potential[slicer],
                    "flux_magnitude": flux_mag[slicer],
                    "se_fraction": se_fraction[slicer],
                },
            }
        )
        print(f"[{sample_key}] loaded {source}: shape={common_shape}, downsample={downsample_factor}")
    return entries


def display_limits(entries: list[dict], field: str, se_threshold: float, lo: float, hi: float) -> tuple[float, float]:
    values = []
    for entry in entries:
        field_arr = entry["fields"][field]
        se_mask = entry["fields"]["se_fraction"] >= se_threshold
        vals = field_arr[np.isfinite(field_arr) & se_mask]
        if vals.size:
            values.append(vals.astype(np.float32, copy=False))
    if not values:
        return 0.0, 1.0
    merged = np.concatenate(values)
    vmin, vmax = np.nanpercentile(merged, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(merged))
        vmax = float(np.nanmax(merged))
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def scaled_voxel_um(cfg: dict, entries: list[dict]) -> dict[str, float]:
    voxel = voxel_size_by_axis(cfg)
    factor = max(int(entry.get("downsample_factor", 1)) for entry in entries) if entries else 1
    return {axis: value * factor for axis, value in voxel.items()}


def plane_extent_um(entries: list[dict], plane: str, voxel_um: dict[str, float]) -> tuple[float, float, float, float]:
    axis_h, axis_v = PLANE_AXES[plane]
    width = 0.0
    height = 0.0
    for entry in entries:
        image = orthogonal_center_slices(entry["fields"]["potential"])[plane]
        width = max(width, image.shape[0] * voxel_um[axis_h])
        height = max(height, image.shape[1] * voxel_um[axis_v])
    return 0.0, width, 0.0, height


def image_extent_um(image: np.ndarray, plane: str, voxel_um: dict[str, float]) -> tuple[float, float, float, float]:
    axis_h, axis_v = PLANE_AXES[plane]
    return 0.0, image.shape[0] * voxel_um[axis_h], 0.0, image.shape[1] * voxel_um[axis_v]


def masked_plane(entry: dict, field: str, plane: str, se_threshold: float) -> np.ma.MaskedArray:
    image = orthogonal_center_slices(entry["fields"][field])[plane]
    se_slice = orthogonal_center_slices(entry["fields"]["se_fraction"])[plane]
    mask = se_slice < se_threshold
    return np.ma.array(image, mask=mask)


def transparent_cmap(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    return cmap


def save_comparison(entries: list[dict], cfg: dict, out: Path, se_threshold: float,
                    percentile_low: float, percentile_high: float, plane: str | None = None) -> None:
    if not entries:
        return

    planes = [plane] if plane else PLANES
    voxel_um = scaled_voxel_um(cfg, entries)
    potential_limits = display_limits(entries, "potential", se_threshold, percentile_low, percentile_high)
    flux_limits = display_limits(entries, "flux_magnitude", se_threshold, percentile_low, percentile_high)
    nrows = 2 * len(planes)
    ncols = len(entries)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.1 * ncols, 3.9 * nrows), squeeze=False,
                             constrained_layout=True)
    cmaps = {
        "potential": transparent_cmap("viridis"),
        "flux_magnitude": transparent_cmap("inferno"),
    }

    for plane_index, plane_name in enumerate(planes):
        axis_h, axis_v = PLANE_AXES[plane_name]
        common_extent = plane_extent_um(entries, plane_name, voxel_um)
        for col, entry in enumerate(entries):
            subtitle = entry["meta"].get("subvolume_id", "whole ROI")
            if entry["source"] == "whole_roi":
                subtitle = "whole ROI"
            for row_offset, field, label, limits in [
                (0, "potential", "SE-only potential / concentration", potential_limits),
                (1, "flux_magnitude", "SE-only flux magnitude", flux_limits),
            ]:
                row = 2 * plane_index + row_offset
                ax = axes[row, col]
                image = masked_plane(entry, field, plane_name, se_threshold)
                im = ax.imshow(
                    image.T,
                    origin="lower",
                    cmap=cmaps[field],
                    interpolation="nearest",
                    vmin=limits[0],
                    vmax=limits[1],
                    extent=image_extent_um(image, plane_name, voxel_um),
                    aspect="equal",
                )
                ax.set_facecolor("#eeeeee")
                ax.set_xlim(common_extent[0], common_extent[1])
                ax.set_ylim(common_extent[2], common_extent[3])
                ax.set_xlabel(f"{axis_h} (um)")
                ax.set_ylabel(f"{axis_v} (um)")
                ax.set_title(f"{entry['sample']} {subtitle}\n{plane_name} {label}")
                if col == ncols - 1:
                    fig.colorbar(im, ax=ax, shrink=0.82, label=label)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, transparent=False)
    plt.close(fig)


def plot_source(cfg: dict, source: str, args: argparse.Namespace) -> None:
    entries = load_entries(cfg, source)
    if not entries:
        print(f"[{source}] no entries to plot.")
        return

    out_dir = root_path(cfg, args.output_dir, source)
    save_comparison(
        entries,
        cfg,
        out_dir / f"{source}_se_only_potential_flux_comparison.png",
        args.se_threshold,
        args.percentile_low,
        args.percentile_high,
    )
    for plane in PLANES:
        save_comparison(
            entries,
            cfg,
            out_dir / f"{source}_se_only_potential_flux_comparison_{plane}.png",
            args.se_threshold,
            args.percentile_low,
            args.percentile_high,
            plane=plane,
        )
    print(f"[{source}] saved SE-only figures in {out_dir}")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    sources = ["representative", "whole_roi"] if args.source == "both" else [args.source]
    for source in sources:
        plot_source(cfg, source, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

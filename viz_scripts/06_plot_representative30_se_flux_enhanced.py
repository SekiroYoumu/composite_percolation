from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

from viz_style import apply_publication_style, panel_figsize, save_figure, style_axes, style_colorbar


SAMPLES = ("WM", "PFDT")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLANES = ("xy_center_z", "xz_center_y", "yz_center_x")
PLANE_AXES = {
    "xy_center_z": ("x", "y"),
    "xz_center_y": ("x", "z"),
    "yz_center_x": ("y", "z"),
}
PLANE_LABELS = {
    "xy_center_z": "xy, center z",
    "xz_center_y": "xz, center y",
    "yz_center_x": "yz, center x",
}


apply_publication_style()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make 30 um representative SE-only potential and flux visualization figures."
    )
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument(
        "--representative-root",
        default=PROJECT_ROOT / "server_results" / "30um-in-plane" / "results",
    )
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "viz" / "representative30_se_only")
    parser.add_argument("--se-label", type=int, default=2)
    parser.add_argument("--se-threshold", type=float, default=0.5)
    parser.add_argument("--voxel-size-um", type=float, default=0.07)
    parser.add_argument("--flux-p-low", type=float, default=5.0)
    parser.add_argument("--flux-p-high", type=float, default=99.0)
    parser.add_argument("--potential-p-low", type=float, default=1.0)
    parser.add_argument("--potential-p-high", type=float, default=99.0)
    parser.add_argument("--top-flux-fractions", type=float, nargs="+", default=[0.20, 0.10])
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def label_path(project_root: Path, sample: str) -> Path:
    candidates = [
        project_root / "intermediate" / "labels" / sample / "label_zyx.tif",
        project_root / "results" / sample / "label_zyx.tif",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find label_zyx.tif for {sample}")


def crop_label_to_metadata(label_zyx: np.ndarray, meta: dict) -> np.ndarray:
    coords = meta.get("coordinates")
    if not coords:
        return label_zyx
    return label_zyx[
        int(coords["z0"]): int(coords["z1"]),
        int(coords["y0"]): int(coords["y1"]),
        int(coords["x0"]): int(coords["x1"]),
    ]


def common_crop(*arrays: np.ndarray) -> list[np.ndarray]:
    common = tuple(min(array.shape[i] for array in arrays) for i in range(3))
    slicer = tuple(slice(0, n) for n in common)
    return [array[slicer] for array in arrays]


def load_entry(project_root: Path, representative_root: Path, sample: str, se_label: int) -> dict:
    field_dir = representative_root / sample / "representative_flux"
    npz_path = field_dir / "median_representative_fields.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    meta = read_json(field_dir / "metadata.json")
    with np.load(npz_path) as data:
        potential_key = "concentration" if "concentration" in data.files else "potential"
        if potential_key not in data.files or "flux_magnitude" not in data.files:
            raise KeyError(f"{npz_path} keys are {data.files}; expected potential/concentration and flux_magnitude.")
        potential = np.asarray(data[potential_key], dtype=np.float32)
        flux = np.asarray(data["flux_magnitude"], dtype=np.float32)

    label_zyx = tiff.imread(label_path(project_root, sample))
    label_xyz = np.transpose(crop_label_to_metadata(label_zyx, meta), (2, 1, 0))
    se_fraction = (label_xyz == se_label).astype(np.float32, copy=False)
    potential, flux, se_fraction = common_crop(potential, flux, se_fraction)
    return {
        "sample": sample,
        "meta": meta,
        "potential": potential,
        "flux_magnitude": flux,
        "se_fraction": se_fraction,
    }


def center_slices_xyz(volume: np.ndarray) -> dict[str, np.ndarray]:
    nx, ny, nz = volume.shape
    return {
        "xy_center_z": volume[:, :, nz // 2],
        "xz_center_y": volume[:, ny // 2, :],
        "yz_center_x": volume[nx // 2, :, :],
    }


def image_extent_um(image: np.ndarray, plane: str, voxel_um: float) -> tuple[float, float, float, float]:
    axis_h, axis_v = PLANE_AXES[plane]
    if plane == "xy_center_z":
        sizes = {"x": image.shape[0], "y": image.shape[1]}
    elif plane == "xz_center_y":
        sizes = {"x": image.shape[0], "z": image.shape[1]}
    else:
        sizes = {"y": image.shape[0], "z": image.shape[1]}
    return 0.0, sizes[axis_h] * voxel_um, 0.0, sizes[axis_v] * voxel_um


def masked_center_slice(entry: dict, field: str, plane: str, se_threshold: float) -> np.ma.MaskedArray:
    image = center_slices_xyz(entry[field])[plane]
    se = center_slices_xyz(entry["se_fraction"])[plane]
    return np.ma.array(image, mask=se < se_threshold)


def transparent_cmap(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad((0.92, 0.92, 0.92, 1.0))
    return cmap


def se_values(entries: list[dict], field: str, se_threshold: float) -> np.ndarray:
    values = []
    for entry in entries:
        arr = entry[field]
        mask = entry["se_fraction"] >= se_threshold
        vals = arr[np.isfinite(arr) & mask]
        if vals.size:
            values.append(vals.astype(np.float32, copy=False))
    return np.concatenate(values) if values else np.array([], dtype=np.float32)


def percentile_limits(values: np.ndarray, low: float, high: float) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(values, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def save_slice_comparison(entries: list[dict], out_dir: Path, plane: str, voxel_um: float,
                          se_threshold: float, pot_limits: tuple[float, float],
                          flux_limits: tuple[float, float], relative_limits: tuple[float, float],
                          global_flux_median: float) -> None:
    fig, axes = plt.subplots(3, len(entries), figsize=panel_figsize(len(entries), 3), constrained_layout=True)
    if len(entries) == 1:
        axes = axes[:, None]
    cmaps = {
        "potential": transparent_cmap("viridis"),
        "flux_magnitude": transparent_cmap("magma"),
        "flux_relative": transparent_cmap("coolwarm"),
    }
    limits_by_field = {
        "potential": pot_limits,
        "flux_magnitude": flux_limits,
        "flux_relative": relative_limits,
    }
    titles = {
        "potential": "Potential / concentration",
        "flux_magnitude": "Flux magnitude",
        "flux_relative": "Flux / median",
    }
    colorbar_labels = {"potential": "potential", "flux_magnitude": "flux", "flux_relative": "flux / median"}
    last_images = {}
    for col, entry in enumerate(entries):
        for row, field in enumerate(("potential", "flux_magnitude", "flux_relative")):
            ax = axes[row, col]
            if field == "flux_relative":
                image = masked_center_slice(entry, "flux_magnitude", plane, se_threshold) / global_flux_median
            else:
                image = masked_center_slice(entry, field, plane, se_threshold)
            im = ax.imshow(
                image.T,
                origin="lower",
                cmap=cmaps[field],
                interpolation="nearest",
                vmin=limits_by_field[field][0],
                vmax=limits_by_field[field][1],
                extent=image_extent_um(image, plane, voxel_um),
                aspect="equal",
            )
            last_images[field] = im
            axis_h, axis_v = PLANE_AXES[plane]
            ax.set_xlabel(f"{axis_h} (um)")
            ax.set_ylabel(f"{axis_v} (um)")
            ax.set_title(f"{entry['sample']} {PLANE_LABELS[plane]}\n{titles[field]}")
    for row, field in enumerate(("potential", "flux_magnitude", "flux_relative")):
        style_colorbar(fig.colorbar(last_images[field], ax=axes[row, :], shrink=0.75, label=colorbar_labels[field]))
    path = out_dir / f"representative30_se_enhanced_{plane}.png"
    style_axes(axes)
    save_figure(fig, path)
    plt.close(fig)
    print(f"Saved {path}")


def save_top_flux_overlay(entries: list[dict], out_dir: Path, plane: str, voxel_um: float,
                          se_threshold: float, top_fractions: list[float]) -> None:
    fig, axes = plt.subplots(1, len(entries), figsize=panel_figsize(len(entries), 1, extra_width_mm=24.0),
                             constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.82, bottom=0.12, top=0.86, wspace=0.28)
    if len(entries) == 1:
        axes = [axes]
    colors = ["#fdae61", "#d7191c", "#7f0000"]
    for ax, entry in zip(axes, entries):
        flux = center_slices_xyz(entry["flux_magnitude"])[plane]
        se = center_slices_xyz(entry["se_fraction"])[plane] >= se_threshold
        base = np.full((*flux.shape, 4), (0.92, 0.92, 0.92, 1.0), dtype=np.float32)
        base[se] = (0.42, 0.42, 0.42, 1.0)
        overlay = base.copy()
        se_flux = flux[np.isfinite(flux) & se]
        for idx, frac in enumerate(sorted(top_fractions, reverse=True)):
            if se_flux.size == 0:
                continue
            threshold = np.nanpercentile(se_flux, 100.0 * (1.0 - frac))
            overlay[se & (flux >= threshold)] = matplotlib.colors.to_rgba(colors[min(idx, len(colors) - 1)], alpha=0.95)
        ax.imshow(
            overlay.transpose(1, 0, 2),
            origin="lower",
            interpolation="nearest",
            extent=image_extent_um(flux, plane, voxel_um),
            aspect="equal",
        )
        axis_h, axis_v = PLANE_AXES[plane]
        ax.set_xlabel(f"{axis_h} (um)")
        ax.set_ylabel(f"{axis_v} (um)")
        ax.set_title(f"{entry['sample']} {PLANE_LABELS[plane]}\nhigh-flux channels")
    handles = [
        plt.Line2D([0], [0], color=(0.42, 0.42, 0.42), lw=8, label="SE phase"),
        plt.Line2D([0], [0], color="#fdae61", lw=8, label=f"top {int(top_fractions[0] * 100)}% SE flux"),
    ]
    if len(top_fractions) > 1:
        handles.append(plt.Line2D([0], [0], color="#d7191c", lw=8, label=f"top {int(top_fractions[1] * 100)}% SE flux"))
    handles.append(plt.Line2D([0], [0], color=(0.92, 0.92, 0.92), lw=8, label="non-SE"))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.84, 0.50), frameon=False)
    path = out_dir / f"representative30_se_top_flux_overlay_{plane}.png"
    style_axes(axes)
    save_figure(fig, path)
    plt.close(fig)
    print(f"Saved {path}")


def save_flux_distribution(entries: list[dict], out_dir: Path, se_threshold: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=panel_figsize(2, 1), constrained_layout=True)
    for entry in entries:
        vals = entry["flux_magnitude"][entry["se_fraction"] >= se_threshold]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            continue
        p1, p999 = np.nanpercentile(vals, [1, 99.9])
        clipped = vals[(vals >= p1) & (vals <= p999)]
        axes[0].hist(clipped, bins=120, density=True, histtype="step", lw=2.0, label=entry["sample"])
        sorted_vals = np.sort(clipped)
        axes[1].plot(sorted_vals, np.linspace(0, 1, sorted_vals.size), lw=2.0, label=entry["sample"])
    axes[0].set_xlabel("Flux magnitude")
    axes[0].set_ylabel("density")
    axes[0].set_title("Flux distribution")
    axes[1].set_xlabel("Flux magnitude")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("Flux CDF")
    for ax in axes:
        ax.legend(loc="upper right")
        style_axes(ax)
    path = out_dir / "representative30_se_flux_distribution.png"
    save_figure(fig, path)
    plt.close(fig)
    print(f"Saved {path}")


def save_y_profiles(entries: list[dict], out_dir: Path, voxel_um: float, se_threshold: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=panel_figsize(2, 1), constrained_layout=True)
    for entry in entries:
        flux = entry["flux_magnitude"]
        potential = entry["potential"]
        se = entry["se_fraction"] >= se_threshold
        y = np.arange(flux.shape[1], dtype=np.float32) * voxel_um
        flux_mean = np.full(flux.shape[1], np.nan, dtype=np.float32)
        flux_p25 = np.full(flux.shape[1], np.nan, dtype=np.float32)
        flux_p75 = np.full(flux.shape[1], np.nan, dtype=np.float32)
        pot_mean = np.full(flux.shape[1], np.nan, dtype=np.float32)
        for yi in range(flux.shape[1]):
            mask = se[:, yi, :]
            vals = flux[:, yi, :][mask]
            pvals = potential[:, yi, :][mask]
            if vals.size:
                flux_mean[yi] = float(np.nanmean(vals))
                flux_p25[yi], flux_p75[yi] = np.nanpercentile(vals, [25, 75])
            if pvals.size:
                pot_mean[yi] = float(np.nanmean(pvals))
        axes[0].plot(y, flux_mean, lw=2.0, label=entry["sample"])
        axes[0].fill_between(y, flux_p25, flux_p75, alpha=0.18)
        axes[1].plot(y, pot_mean, lw=2.0, label=entry["sample"])
    axes[0].set_xlabel("y (um)")
    axes[0].set_ylabel("Flux magnitude")
    axes[0].set_title("Plane-wise flux")
    axes[1].set_xlabel("y (um)")
    axes[1].set_ylabel("Potential / concentration")
    axes[1].set_title("Plane-wise potential")
    for ax in axes:
        ax.legend(loc="upper right")
        style_axes(ax)
    path = out_dir / "representative30_se_y_profiles.png"
    save_figure(fig, path)
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    representative_root = Path(args.representative_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = [load_entry(project_root, representative_root, sample, args.se_label) for sample in SAMPLES]
    se_flux_values = se_values(entries, "flux_magnitude", args.se_threshold)
    se_potential_values = se_values(entries, "potential", args.se_threshold)
    flux_limits = percentile_limits(se_flux_values, args.flux_p_low, args.flux_p_high)
    pot_limits = percentile_limits(se_potential_values, args.potential_p_low, args.potential_p_high)
    global_flux_median = float(np.nanmedian(se_flux_values[se_flux_values > 0]))
    relative_limits = percentile_limits(se_flux_values / global_flux_median, 5.0, 99.0)
    print(f"Shared potential limits: {pot_limits}")
    print(f"Shared flux limits: {flux_limits}")
    print(f"Global SE flux median: {global_flux_median:.6g}; relative limits: {relative_limits}")
    for plane in PLANES:
        save_slice_comparison(entries, out_dir, plane, args.voxel_size_um, args.se_threshold, pot_limits,
                              flux_limits, relative_limits, global_flux_median)
        save_top_flux_overlay(entries, out_dir, plane, args.voxel_size_um, args.se_threshold, args.top_flux_fractions)
    save_flux_distribution(entries, out_dir, args.se_threshold)
    save_y_profiles(entries, out_dir, args.voxel_size_um, args.se_threshold)


if __name__ == "__main__":
    main()

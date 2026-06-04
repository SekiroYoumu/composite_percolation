from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff


SAMPLES = ("WM", "PFDT")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLANES = ("xy_center_z", "xz_center_y", "yz_center_x")
PLANE_AXES = {
    "xy_center_z": ("x", "y"),
    "xz_center_y": ("x", "z"),
    "yz_center_x": ("y", "z"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make whole-ROI SE-only visualization figures with shared color scales. "
            "The script masks non-SE voxels and emphasizes flux heterogeneity inside the SE phase."
        )
    )
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument(
        "--whole-roi-root",
        default=PROJECT_ROOT / "server_results" / "whole_roi",
        help="Folder containing bulk_fields/<sample>/fields/whole_roi/*.npz.",
    )
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "viz" / "whole_roi_se_only")
    parser.add_argument("--se-label", type=int, default=2)
    parser.add_argument("--voxel-size-um", type=float, default=0.07)
    parser.add_argument("--downsample-factor", type=int, default=None)
    parser.add_argument("--se-threshold", type=float, default=0.5)
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


def load_fields(whole_roi_root: Path, sample: str) -> tuple[np.ndarray, np.ndarray, dict]:
    field_dir = whole_roi_root / "bulk_fields" / sample / "fields" / "whole_roi"
    npz_path = field_dir / "whole_roi_downsampled_fields.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing field file: {npz_path}")
    meta = read_json(field_dir / "metadata.json")
    with np.load(npz_path) as data:
        potential_key = "concentration" if "concentration" in data.files else "potential"
        if potential_key not in data.files or "flux_magnitude" not in data.files:
            raise KeyError(f"{npz_path} keys are {data.files}; expected concentration/potential and flux_magnitude.")
        potential = np.asarray(data[potential_key], dtype=np.float32)
        flux = np.asarray(data["flux_magnitude"], dtype=np.float32)
    common = tuple(min(a, b) for a, b in zip(potential.shape, flux.shape))
    slicer = tuple(slice(0, n) for n in common)
    return potential[slicer], flux[slicer], meta


def label_path(project_root: Path, sample: str) -> Path:
    candidates = [
        project_root / "intermediate" / "labels" / sample / "label_zyx.tif",
        project_root / "results" / sample / "label_zyx.tif",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find label_zyx.tif for {sample}. Checked: {candidates}")


def transpose_zyx_to_xyz(array: np.ndarray) -> np.ndarray:
    return np.transpose(array, (2, 1, 0))


def downsample_binary_fraction(mask_xyz: np.ndarray, factor: int, target_shape: tuple[int, int, int]) -> np.ndarray:
    if factor <= 1:
        common = tuple(min(a, b) for a, b in zip(mask_xyz.shape, target_shape))
        return mask_xyz[tuple(slice(0, n) for n in common)].astype(np.float32, copy=False)

    crop_shape = tuple(min(mask_xyz.shape[i], target_shape[i] * factor) for i in range(3))
    crop_shape = tuple((n // factor) * factor for n in crop_shape)
    cropped = mask_xyz[tuple(slice(0, n) for n in crop_shape)].astype(np.float32, copy=False)
    reshaped = cropped.reshape(
        crop_shape[0] // factor,
        factor,
        crop_shape[1] // factor,
        factor,
        crop_shape[2] // factor,
        factor,
    )
    fraction = reshaped.mean(axis=(1, 3, 5), dtype=np.float32)
    common = tuple(min(a, b) for a, b in zip(fraction.shape, target_shape))
    return fraction[tuple(slice(0, n) for n in common)]


def load_se_fraction(project_root: Path, sample: str, se_label: int, factor: int,
                     target_shape: tuple[int, int, int]) -> np.ndarray:
    label_zyx = tiff.imread(label_path(project_root, sample))
    label_xyz = transpose_zyx_to_xyz(label_zyx)
    return downsample_binary_fraction(label_xyz == se_label, factor, target_shape)


def center_slices_xyz(volume: np.ndarray) -> dict[str, np.ndarray]:
    nx, ny, nz = volume.shape
    return {
        "xy_center_z": volume[:, :, nz // 2],
        "xz_center_y": volume[:, ny // 2, :],
        "yz_center_x": volume[nx // 2, :, :],
    }


def masked_center_slice(entry: dict, field: str, plane: str, se_threshold: float) -> np.ma.MaskedArray:
    image = center_slices_xyz(entry[field])[plane]
    se = center_slices_xyz(entry["se_fraction"])[plane]
    return np.ma.array(image, mask=se < se_threshold)


def image_extent_um(image: np.ndarray, plane: str, voxel_um: float) -> tuple[float, float, float, float]:
    axis_h, axis_v = PLANE_AXES[plane]
    axis_to_size = {"x": image.shape[0], "y": image.shape[1]}
    if plane == "xz_center_y":
        axis_to_size = {"x": image.shape[0], "z": image.shape[1]}
    elif plane == "yz_center_x":
        axis_to_size = {"y": image.shape[0], "z": image.shape[1]}
    return 0.0, axis_to_size[axis_h] * voxel_um, 0.0, axis_to_size[axis_v] * voxel_um


def se_values(entries: list[dict], field: str, se_threshold: float) -> np.ndarray:
    values = []
    for entry in entries:
        mask = entry["se_fraction"] >= se_threshold
        arr = entry[field]
        vals = arr[np.isfinite(arr) & mask]
        if vals.size:
            values.append(vals.astype(np.float32, copy=False))
    if not values:
        return np.array([], dtype=np.float32)
    return np.concatenate(values)


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


def transparent_cmap(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad((0.92, 0.92, 0.92, 1.0))
    return cmap


def save_slice_comparison(entries: list[dict], out_dir: Path, plane: str, voxel_um: float,
                          se_threshold: float, pot_limits: tuple[float, float],
                          flux_limits: tuple[float, float], relative_limits: tuple[float, float],
                          global_flux_median: float) -> None:
    fig, axes = plt.subplots(3, len(entries), figsize=(5.6 * len(entries), 13.0), constrained_layout=True)
    if len(entries) == 1:
        axes = axes[:, None]

    potential_cmap = transparent_cmap("viridis")
    flux_cmap = transparent_cmap("magma")
    rel_cmap = transparent_cmap("coolwarm")
    last_images = {}

    for col, entry in enumerate(entries):
        for row, (field, title, cmap, limits) in enumerate(
            [
                ("potential", "SE-only potential / concentration", potential_cmap, pot_limits),
                ("flux_magnitude", "SE-only flux magnitude, percentile-clipped", flux_cmap, flux_limits),
                ("flux_relative", "SE-only flux / global SE median", rel_cmap, relative_limits),
            ]
        ):
            ax = axes[row, col]
            if field == "flux_relative":
                image = masked_center_slice(entry, "flux_magnitude", plane, se_threshold) / global_flux_median
            else:
                image = masked_center_slice(entry, field, plane, se_threshold)
            im = ax.imshow(
                image.T,
                origin="lower",
                cmap=cmap,
                interpolation="nearest",
                vmin=limits[0],
                vmax=limits[1],
                extent=image_extent_um(image, plane, voxel_um),
                aspect="equal",
            )
            last_images[field] = im
            axis_h, axis_v = PLANE_AXES[plane]
            ax.set_xlabel(f"{axis_h} (um)")
            ax.set_ylabel(f"{axis_v} (um)")
            ax.set_title(f"{entry['sample']} {plane}\n{title}")

    labels = {
        "potential": "potential / concentration",
        "flux_magnitude": "flux magnitude",
        "flux_relative": "flux / global SE median",
    }
    for row, field in enumerate(("potential", "flux_magnitude", "flux_relative")):
        fig.colorbar(last_images[field], ax=axes[row, :], shrink=0.75, label=labels[field])

    path = out_dir / f"whole_roi_se_enhanced_{plane}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved {path}")


def save_top_flux_overlay(entries: list[dict], out_dir: Path, plane: str, voxel_um: float,
                          se_threshold: float, top_fractions: list[float]) -> None:
    fig, axes = plt.subplots(1, len(entries), figsize=(6.2 * len(entries), 5.4), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.82, bottom=0.12, top=0.86, wspace=0.28)
    if len(entries) == 1:
        axes = [axes]
    colors = ["#fdae61", "#d7191c", "#7f0000"]

    for ax, entry in zip(axes, entries):
        flux = center_slices_xyz(entry["flux_magnitude"])[plane]
        se = center_slices_xyz(entry["se_fraction"])[plane] >= se_threshold
        base = np.full((*flux.shape, 4), (0.92, 0.92, 0.92, 1.0), dtype=np.float32)
        base[se] = (0.42, 0.42, 0.42, 1.0)
        se_flux = flux[np.isfinite(flux) & se]
        overlay = base.copy()
        for idx, frac in enumerate(sorted(top_fractions, reverse=True)):
            if se_flux.size == 0:
                continue
            threshold = np.nanpercentile(se_flux, 100.0 * (1.0 - frac))
            rgba = matplotlib.colors.to_rgba(colors[min(idx, len(colors) - 1)], alpha=0.95)
            overlay[se & (flux >= threshold)] = rgba
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
        ax.set_title(f"{entry['sample']} {plane}\nSE high-flux channels")

    handles = [
        plt.Line2D([0], [0], color=(0.42, 0.42, 0.42), lw=8, label="SE phase"),
        plt.Line2D([0], [0], color="#fdae61", lw=8, label=f"top {int(top_fractions[0] * 100)}% SE flux"),
    ]
    if len(top_fractions) > 1:
        handles.append(plt.Line2D([0], [0], color="#d7191c", lw=8, label=f"top {int(top_fractions[1] * 100)}% SE flux"))
    handles.append(plt.Line2D([0], [0], color=(0.92, 0.92, 0.92), lw=8, label="non-SE / low-SE"))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.84, 0.50), frameon=False)
    path = out_dir / f"whole_roi_se_top_flux_overlay_{plane}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved {path}")


def save_flux_distribution(entries: list[dict], out_dir: Path, se_threshold: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for entry in entries:
        vals = entry["flux_magnitude"][entry["se_fraction"] >= se_threshold]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            continue
        p1, p999 = np.nanpercentile(vals, [1, 99.9])
        clipped = vals[(vals >= p1) & (vals <= p999)]
        axes[0].hist(clipped, bins=120, density=True, histtype="step", lw=2.0, label=entry["sample"])
        sorted_vals = np.sort(clipped)
        cdf = np.linspace(0, 1, sorted_vals.size)
        axes[1].plot(sorted_vals, cdf, lw=2.0, label=entry["sample"])
    axes[0].set_xlabel("SE-only flux magnitude")
    axes[0].set_ylabel("density")
    axes[0].set_title("Flux distribution, clipped 1-99.9%")
    axes[1].set_xlabel("SE-only flux magnitude")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("Flux cumulative distribution")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
    path = out_dir / "whole_roi_se_flux_distribution.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved {path}")


def save_y_profiles(entries: list[dict], out_dir: Path, voxel_um: float, se_threshold: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
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
    axes[0].set_xlabel("y (um), transport direction")
    axes[0].set_ylabel("SE-only flux magnitude")
    axes[0].set_title("Plane-wise SE flux: mean and IQR")
    axes[1].set_xlabel("y (um), transport direction")
    axes[1].set_ylabel("SE-only potential / concentration")
    axes[1].set_title("Plane-wise SE potential mean")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
    path = out_dir / "whole_roi_se_y_profiles.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    whole_roi_root = Path(args.whole_roi_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for sample in SAMPLES:
        potential, flux, meta = load_fields(whole_roi_root, sample)
        downsample = args.downsample_factor or int(meta.get("downsample_factor", 2))
        se_fraction = load_se_fraction(project_root, sample, args.se_label, downsample, potential.shape)
        common = tuple(min(potential.shape[i], flux.shape[i], se_fraction.shape[i]) for i in range(3))
        slicer = tuple(slice(0, n) for n in common)
        entry = {
            "sample": sample,
            "potential": potential[slicer],
            "flux_magnitude": flux[slicer],
            "se_fraction": se_fraction[slicer],
            "downsample_factor": downsample,
        }
        entries.append(entry)
        vals = entry["flux_magnitude"][entry["se_fraction"] >= args.se_threshold]
        print(
            f"{sample}: shape={common}, downsample={downsample}, "
            f"SE voxels={vals.size}, SE flux median={np.nanmedian(vals):.6g}, mean={np.nanmean(vals):.6g}"
        )

    voxel_um = float(args.voxel_size_um) * max(entry["downsample_factor"] for entry in entries)
    se_flux_values = se_values(entries, "flux_magnitude", args.se_threshold)
    se_potential_values = se_values(entries, "potential", args.se_threshold)
    flux_limits = percentile_limits(se_flux_values, args.flux_p_low, args.flux_p_high)
    pot_limits = percentile_limits(se_potential_values, args.potential_p_low, args.potential_p_high)
    global_flux_median = float(np.nanmedian(se_flux_values[se_flux_values > 0]))
    relative_values = se_flux_values / global_flux_median
    relative_limits = percentile_limits(relative_values, 5.0, 99.0)
    print(f"Shared potential limits: {pot_limits}")
    print(f"Shared flux limits: {flux_limits}")
    print(f"Global SE flux median: {global_flux_median:.6g}; relative limits: {relative_limits}")

    for plane in PLANES:
        save_slice_comparison(entries, out_dir, plane, voxel_um, args.se_threshold, pot_limits,
                              flux_limits, relative_limits, global_flux_median)
        save_top_flux_overlay(entries, out_dir, plane, voxel_um, args.se_threshold, args.top_flux_fractions)
    save_flux_distribution(entries, out_dir, args.se_threshold)
    save_y_profiles(entries, out_dir, voxel_um, args.se_threshold)


if __name__ == "__main__":
    main()

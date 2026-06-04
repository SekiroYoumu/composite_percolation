from __future__ import annotations

import argparse
import csv
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
            "Quantify SE-only flux heterogeneity and bottleneck-like current localization "
            "from whole-ROI downsampled fields."
        )
    )
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument("--whole-roi-root", default=PROJECT_ROOT / "server_results" / "whole_roi")
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "viz" / "whole_roi_flux_heterogeneity")
    parser.add_argument("--se-label", type=int, default=2)
    parser.add_argument("--se-threshold", type=float, default=0.5)
    parser.add_argument("--voxel-size-um", type=float, default=0.07)
    parser.add_argument("--downsample-factor", type=int, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_fields(whole_roi_root: Path, sample: str) -> tuple[np.ndarray, dict]:
    field_dir = whole_roi_root / "bulk_fields" / sample / "fields" / "whole_roi"
    npz_path = field_dir / "whole_roi_downsampled_fields.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path) as data:
        if "flux_magnitude" not in data.files:
            raise KeyError(f"{npz_path} keys are {data.files}; expected flux_magnitude.")
        flux = np.asarray(data["flux_magnitude"], dtype=np.float32)
    return flux, read_json(field_dir / "metadata.json")


def label_path(project_root: Path, sample: str) -> Path:
    candidates = [
        project_root / "intermediate" / "labels" / sample / "label_zyx.tif",
        project_root / "results" / sample / "label_zyx.tif",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find label_zyx.tif for {sample}")


def downsample_se_fraction(label_zyx: np.ndarray, se_label: int, factor: int,
                           target_shape: tuple[int, int, int]) -> np.ndarray:
    label_xyz = np.transpose(label_zyx, (2, 1, 0))
    se = label_xyz == se_label
    if factor <= 1:
        common = tuple(min(a, b) for a, b in zip(se.shape, target_shape))
        return se[tuple(slice(0, n) for n in common)].astype(np.float32, copy=False)
    crop_shape = tuple(min(se.shape[i], target_shape[i] * factor) for i in range(3))
    crop_shape = tuple((n // factor) * factor for n in crop_shape)
    cropped = se[tuple(slice(0, n) for n in crop_shape)].astype(np.float32, copy=False)
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


def load_entry(project_root: Path, whole_roi_root: Path, sample: str, se_label: int,
               se_threshold: float, downsample_arg: int | None) -> dict:
    flux, meta = load_fields(whole_roi_root, sample)
    downsample = downsample_arg or int(meta.get("downsample_factor", 2))
    label_zyx = tiff.imread(label_path(project_root, sample))
    se_fraction = downsample_se_fraction(label_zyx, se_label, downsample, flux.shape)
    common = tuple(min(flux.shape[i], se_fraction.shape[i]) for i in range(3))
    slicer = tuple(slice(0, n) for n in common)
    flux = flux[slicer]
    se_fraction = se_fraction[slicer]
    se_mask = se_fraction >= se_threshold
    vals = flux[np.isfinite(flux) & se_mask].astype(np.float32, copy=False)
    vals = vals[vals >= 0]
    return {
        "sample": sample,
        "flux": flux,
        "se_fraction": se_fraction,
        "se_mask": se_mask,
        "values": vals,
        "downsample_factor": downsample,
    }


def gini(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    values = values[values >= 0]
    if values.size == 0:
        return float("nan")
    sorted_values = np.sort(values.astype(np.float64, copy=False))
    total = sorted_values.sum()
    if total <= 0:
        return 0.0
    n = sorted_values.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_values) / (n * total)) - ((n + 1.0) / n))


def top_share(values: np.ndarray, fraction: float) -> float:
    if values.size == 0:
        return float("nan")
    threshold = np.nanpercentile(values, 100.0 * (1.0 - fraction))
    total = float(np.nansum(values))
    if total <= 0:
        return float("nan")
    return float(np.nansum(values[values >= threshold]) / total)


def bottom_share(values: np.ndarray, fraction: float) -> float:
    if values.size == 0:
        return float("nan")
    threshold = np.nanpercentile(values, 100.0 * fraction)
    total = float(np.nansum(values))
    if total <= 0:
        return float("nan")
    return float(np.nansum(values[values <= threshold]) / total)


def participation_ratio(values: np.ndarray) -> float:
    values = values[np.isfinite(values)].astype(np.float64, copy=False)
    values = values[values >= 0]
    if values.size == 0:
        return float("nan")
    numerator = float(np.sum(values) ** 2)
    denominator = float(values.size * np.sum(values ** 2))
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def plane_profile(entry: dict) -> dict[str, np.ndarray]:
    flux = entry["flux"]
    se = entry["se_mask"]
    ny = flux.shape[1]
    mean = np.full(ny, np.nan, dtype=np.float32)
    total = np.full(ny, np.nan, dtype=np.float32)
    p10 = np.full(ny, np.nan, dtype=np.float32)
    p90 = np.full(ny, np.nan, dtype=np.float32)
    area = np.full(ny, 0, dtype=np.int64)
    for y in range(ny):
        mask = se[:, y, :]
        vals = flux[:, y, :][mask]
        vals = vals[np.isfinite(vals)]
        area[y] = vals.size
        if vals.size:
            mean[y] = float(np.nanmean(vals))
            total[y] = float(np.nansum(vals))
            p10[y], p90[y] = np.nanpercentile(vals, [10, 90])
    return {"mean": mean, "total": total, "p10": p10, "p90": p90, "area": area}


def metrics_for_entry(entry: dict, global_median: float) -> dict[str, float | str | int]:
    values = entry["values"]
    profile = plane_profile(entry)
    finite_mean = profile["mean"][np.isfinite(profile["mean"])]
    finite_total = profile["total"][np.isfinite(profile["total"])]
    p = np.nanpercentile(values, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    median = float(p[4])
    plane_mean_median = float(np.nanmedian(finite_mean))
    plane_total_median = float(np.nanmedian(finite_total))
    min_plane_idx = int(np.nanargmin(profile["mean"]))
    return {
        "sample": entry["sample"],
        "n_se_voxels": int(values.size),
        "mean": mean,
        "median": median,
        "std": std,
        "cv": std / mean if mean > 0 else float("nan"),
        "gini": gini(values),
        "participation_ratio": participation_ratio(values),
        "localization_index_1_minus_participation": 1.0 - participation_ratio(values),
        "top_10pct_flux_share": top_share(values, 0.10),
        "top_20pct_flux_share": top_share(values, 0.20),
        "bottom_50pct_flux_share": bottom_share(values, 0.50),
        "p01": float(p[0]),
        "p05": float(p[1]),
        "p10": float(p[2]),
        "p25": float(p[3]),
        "p50": median,
        "p75": float(p[5]),
        "p90": float(p[6]),
        "p95": float(p[7]),
        "p99": float(p[8]),
        "p90_over_p10": float(p[6] / p[2]) if p[2] > 0 else float("nan"),
        "p99_over_p50": float(p[8] / median) if median > 0 else float("nan"),
        "iqr_over_median": float((p[5] - p[3]) / median) if median > 0 else float("nan"),
        "fraction_below_global_median": float(np.mean(values < global_median)),
        "fraction_below_half_global_median": float(np.mean(values < 0.5 * global_median)),
        "fraction_above_2x_global_median": float(np.mean(values > 2.0 * global_median)),
        "plane_mean_cv_y": float(np.nanstd(finite_mean) / np.nanmean(finite_mean)),
        "plane_total_cv_y": float(np.nanstd(finite_total) / np.nanmean(finite_total)),
        "min_plane_mean_over_median_y": float(np.nanmin(finite_mean) / plane_mean_median),
        "min_plane_total_over_median_y": float(np.nanmin(finite_total) / plane_total_median),
        "bottleneck_index_y": float(1.0 - np.nanmin(finite_mean) / plane_mean_median),
        "min_plane_y_index": min_plane_idx,
    }


def save_metrics_csv(rows: list[dict], out_dir: Path) -> None:
    path = out_dir / "whole_roi_flux_heterogeneity_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def save_bar_summary(rows: list[dict], out_dir: Path) -> None:
    metrics = [
        ("cv", "CV"),
        ("gini", "Gini"),
        ("top_10pct_flux_share", "top 10% flux share"),
        ("localization_index_1_minus_participation", "1 - participation ratio"),
        ("bottleneck_index_y", "y bottleneck index"),
    ]
    samples = [row["sample"] for row in rows]
    x = np.arange(len(metrics), dtype=np.float32)
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    for i, row in enumerate(rows):
        vals = [float(row[key]) for key, _ in metrics]
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=samples[i])
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], rotation=20, ha="right")
    ax.set_ylabel("heterogeneity / localization metric")
    ax.set_title("SE-only flux heterogeneity metrics")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    path = out_dir / "whole_roi_flux_heterogeneity_metrics.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved {path}")


def save_lorenz(entries: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
    for entry in entries:
        vals = np.sort(entry["values"].astype(np.float64, copy=False))
        total = vals.sum()
        if total <= 0:
            continue
        cum_flux = np.concatenate([[0.0], np.cumsum(vals) / total])
        cum_voxels = np.linspace(0.0, 1.0, cum_flux.size)
        ax.plot(cum_voxels, cum_flux, lw=2.4, label=entry["sample"])
    ax.plot([0, 1], [0, 1], color="0.6", lw=1.2, ls="--", label="uniform")
    ax.set_xlabel("cumulative fraction of SE voxels, sorted by flux")
    ax.set_ylabel("cumulative fraction of total SE flux")
    ax.set_title("Lorenz curve of SE-only flux localization")
    ax.legend()
    ax.grid(alpha=0.25)
    path = out_dir / "whole_roi_flux_lorenz_curve.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved {path}")


def save_y_profiles(entries: list[dict], out_dir: Path, voxel_um: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for entry in entries:
        profile = plane_profile(entry)
        y = np.arange(profile["mean"].size, dtype=np.float32) * voxel_um
        mean = profile["mean"]
        total = profile["total"]
        axes[0].plot(y, mean / np.nanmedian(mean), lw=2.2, label=entry["sample"])
        axes[1].plot(y, total / np.nanmedian(total), lw=2.2, label=entry["sample"])
    axes[0].set_xlabel("y (um), transport direction")
    axes[0].set_ylabel("plane mean flux / median")
    axes[0].set_title("Relative plane-wise SE flux")
    axes[1].set_xlabel("y (um), transport direction")
    axes[1].set_ylabel("plane total flux / median")
    axes[1].set_title("Relative plane-wise total SE flux")
    for ax in axes:
        ax.axhline(1.0, color="0.5", lw=1.0, ls="--")
        ax.legend()
        ax.grid(alpha=0.25)
    path = out_dir / "whole_roi_flux_bottleneck_y_profiles.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved {path}")


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


def save_low_high_maps(entries: list[dict], out_dir: Path, voxel_um: float, global_median: float) -> None:
    for plane in PLANES:
        fig, axes = plt.subplots(1, len(entries), figsize=(6.1 * len(entries), 5.4), constrained_layout=False)
        fig.subplots_adjust(left=0.07, right=0.82, bottom=0.12, top=0.86, wspace=0.28)
        if len(entries) == 1:
            axes = [axes]
        for ax, entry in zip(axes, entries):
            flux = center_slices_xyz(entry["flux"])[plane]
            se = center_slices_xyz(entry["se_mask"])[plane]
            image = np.full((*flux.shape, 4), (0.92, 0.92, 0.92, 1.0), dtype=np.float32)
            image[se] = (0.64, 0.64, 0.64, 1.0)
            low = se & (flux < 0.5 * global_median)
            high = se & (flux > 2.0 * global_median)
            very_high = se & (flux > np.nanpercentile(entry["values"], 90.0))
            image[low] = matplotlib.colors.to_rgba("#4575b4", alpha=0.95)
            image[high] = matplotlib.colors.to_rgba("#f46d43", alpha=0.95)
            image[very_high] = matplotlib.colors.to_rgba("#d73027", alpha=0.95)
            ax.imshow(
                image.transpose(1, 0, 2),
                origin="lower",
                interpolation="nearest",
                extent=image_extent_um(flux, plane, voxel_um),
                aspect="equal",
            )
            axis_h, axis_v = PLANE_AXES[plane]
            ax.set_xlabel(f"{axis_h} (um)")
            ax.set_ylabel(f"{axis_v} (um)")
            ax.set_title(f"{entry['sample']} {plane}\nunder-used SE and concentrated flux")
        handles = [
            plt.Line2D([0], [0], color="#4575b4", lw=8, label="< 0.5x global median"),
            plt.Line2D([0], [0], color=(0.64, 0.64, 0.64), lw=8, label="SE background"),
            plt.Line2D([0], [0], color="#f46d43", lw=8, label="> 2x global median"),
            plt.Line2D([0], [0], color="#d73027", lw=8, label="sample top 10%"),
            plt.Line2D([0], [0], color=(0.92, 0.92, 0.92), lw=8, label="non-SE / low-SE"),
        ]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.84, 0.50), frameon=False)
        path = out_dir / f"whole_roi_flux_low_high_map_{plane}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        print(f"Saved {path}")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    whole_roi_root = Path(args.whole_roi_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        load_entry(project_root, whole_roi_root, sample, args.se_label, args.se_threshold, args.downsample_factor)
        for sample in SAMPLES
    ]
    all_values = np.concatenate([entry["values"] for entry in entries])
    global_median = float(np.nanmedian(all_values[all_values > 0]))
    rows = [metrics_for_entry(entry, global_median) for entry in entries]
    voxel_um = float(args.voxel_size_um) * max(entry["downsample_factor"] for entry in entries)

    save_metrics_csv(rows, out_dir)
    save_bar_summary(rows, out_dir)
    save_lorenz(entries, out_dir)
    save_y_profiles(entries, out_dir, voxel_um)
    save_low_high_maps(entries, out_dir, voxel_um, global_median)

    print("\nKey bottleneck/localization metrics:")
    for row in rows:
        print(
            f"{row['sample']}: CV={row['cv']:.3f}, Gini={row['gini']:.3f}, "
            f"top10_share={row['top_10pct_flux_share']:.3f}, "
            f"participation={row['participation_ratio']:.3f}, "
            f"y_bottleneck={row['bottleneck_index_y']:.3f}"
        )


if __name__ == "__main__":
    main()

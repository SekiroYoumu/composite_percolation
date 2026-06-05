from __future__ import annotations

import argparse
import csv
import gc
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
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
QUANTITY_LABELS = {
    "flux_magnitude": "|J|",
    "abs_Jx": "|Jx|",
    "abs_Jy": "|Jy|",
    "abs_Jz": "|Jz|",
}


apply_publication_style()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify SE-only flux heterogeneity for the 30 um median-representative "
            "high-resolution PuMA fields."
        )
    )
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument(
        "--representative-root",
        default=PROJECT_ROOT / "server_results" / "30um-in-plane" / "results",
    )
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "viz" / "representative30_flux_heterogeneity")
    parser.add_argument("--se-label", type=int, default=2)
    parser.add_argument("--se-threshold", type=float, default=0.5)
    parser.add_argument("--voxel-size-um", type=float, default=0.07)
    parser.add_argument("--direction", default=None, choices=["x", "y", "z"])
    parser.add_argument("--interior-margin-fraction", type=float, default=0.05)
    parser.add_argument("--potential-p-low", type=float, default=1.0)
    parser.add_argument("--potential-p-high", type=float, default=99.0)
    parser.add_argument("--flux-p-low", type=float, default=1.0)
    parser.add_argument("--flux-p-high", type=float, default=99.0)
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


def crop_label_to_representative(project_root: Path, sample: str, meta: dict) -> np.ndarray:
    label_zyx = tiff.imread(label_path(project_root, sample))
    coords = meta.get("coordinates")
    if coords:
        label_zyx = label_zyx[
            int(coords["z0"]): int(coords["z1"]),
            int(coords["y0"]): int(coords["y1"]),
            int(coords["x0"]): int(coords["x1"]),
        ]
    return np.transpose(label_zyx, (2, 1, 0))


def common_crop(*arrays: np.ndarray) -> list[np.ndarray]:
    common = tuple(min(array.shape[i] for array in arrays) for i in range(3))
    slicer = tuple(slice(0, n) for n in common)
    out = []
    for array in arrays:
        if array.ndim == 4:
            out.append(array[slicer + (slice(None),)])
        else:
            out.append(array[slicer])
    return out


def load_sample_fields(project_root: Path, representative_root: Path, sample: str, se_label: int,
                       se_threshold: float, direction_override: str | None) -> dict:
    field_dir = representative_root / sample / "representative_flux"
    npz_path = field_dir / "median_representative_fields.npz"
    meta_path = field_dir / "metadata.json"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    meta = read_json(meta_path)
    direction = direction_override or meta.get("direction", "y")
    if direction not in AXIS_INDEX:
        raise ValueError(f"Invalid direction {direction!r}; expected x/y/z.")

    with np.load(npz_path) as data:
        potential_key = "concentration" if "concentration" in data.files else "potential"
        if potential_key not in data.files:
            raise KeyError(f"{npz_path} missing concentration/potential; keys={data.files}")
        if "flux_magnitude" not in data.files:
            raise KeyError(f"{npz_path} missing flux_magnitude; keys={data.files}")
        potential = np.asarray(data[potential_key], dtype=np.float32)
        flux_magnitude = np.asarray(data["flux_magnitude"], dtype=np.float32)
        flux_vector = None
        if "flux" in data.files:
            flux_vector = np.asarray(data["flux"], dtype=np.float32)

    label_xyz = crop_label_to_representative(project_root, sample, meta)
    if flux_vector is not None:
        potential, flux_magnitude, label_xyz, flux_vector = common_crop(potential, flux_magnitude, label_xyz, flux_vector)
        transport_abs = np.abs(flux_vector[..., AXIS_INDEX[direction]]).astype(np.float32, copy=False)
        del flux_vector
    else:
        potential, flux_magnitude, label_xyz = common_crop(potential, flux_magnitude, label_xyz)
        transport_abs = None

    se_mask = label_xyz == se_label
    if se_threshold > 0.5:
        se_mask = se_mask.astype(np.float32) >= se_threshold
    quantities = {"flux_magnitude": flux_magnitude}
    if transport_abs is not None:
        quantities[f"abs_J{direction}"] = transport_abs

    print(
        f"{sample}: field_shape={flux_magnitude.shape}, SE voxels={int(np.count_nonzero(se_mask))}, "
        f"direction={direction}, subvolume={meta.get('subvolume_id')}, keff={meta.get('keff_norm')}"
    )
    return {
        "sample": sample,
        "meta": meta,
        "direction": direction,
        "se_mask": se_mask,
        "potential": potential,
        "quantities": quantities,
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
    denominator = float(values.size * np.sum(values ** 2))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(values) ** 2 / denominator)


def plane_profile(scalar: np.ndarray, se_mask: np.ndarray) -> dict[str, np.ndarray]:
    ny = scalar.shape[1]
    mean = np.full(ny, np.nan, dtype=np.float32)
    total = np.full(ny, np.nan, dtype=np.float32)
    p10 = np.full(ny, np.nan, dtype=np.float32)
    p90 = np.full(ny, np.nan, dtype=np.float32)
    area = np.zeros(ny, dtype=np.int64)
    for yi in range(ny):
        mask = se_mask[:, yi, :]
        vals = scalar[:, yi, :][mask]
        vals = vals[np.isfinite(vals)]
        area[yi] = vals.size
        if vals.size:
            mean[yi] = float(np.nanmean(vals))
            total[yi] = float(np.nansum(vals))
            p10[yi], p90[yi] = np.nanpercentile(vals, [10, 90])
    return {"mean": mean, "total": total, "p10": p10, "p90": p90, "area": area}


def interior_slice(n: int, margin_fraction: float) -> slice:
    margin = int(round(n * margin_fraction))
    margin = min(max(margin, 0), max(0, (n - 1) // 2))
    return slice(margin, n - margin if margin else n)


def metrics_for_values(sample: str, quantity: str, scalar: np.ndarray, se_mask: np.ndarray,
                       global_median: float, margin_fraction: float, meta: dict) -> dict[str, float | str | int]:
    values = scalar[np.isfinite(scalar) & se_mask].astype(np.float32, copy=False)
    values = values[values >= 0]
    p = np.nanpercentile(values, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    median = float(p[4])
    profile = plane_profile(scalar, se_mask)
    mean_profile = profile["mean"]
    total_profile = profile["total"]
    mean_valid = mean_profile[np.isfinite(mean_profile)]
    total_valid = total_profile[np.isfinite(total_profile)]
    interior = interior_slice(mean_profile.size, margin_fraction)
    mean_interior = mean_profile[interior]
    total_interior = total_profile[interior]
    mean_interior = mean_interior[np.isfinite(mean_interior)]
    total_interior = total_interior[np.isfinite(total_interior)]
    min_plane_idx = int(np.nanargmin(mean_profile))
    return {
        "sample": sample,
        "subvolume_id": meta.get("subvolume_id", ""),
        "keff_norm": meta.get("keff_norm", ""),
        "group_median_keff_norm": meta.get("group_median_keff_norm", ""),
        "quantity": quantity,
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
        "plane_mean_cv_y": float(np.nanstd(mean_valid) / np.nanmean(mean_valid)),
        "plane_total_cv_y": float(np.nanstd(total_valid) / np.nanmean(total_valid)),
        "bottleneck_index_y_all": float(1.0 - np.nanmin(mean_valid) / np.nanmedian(mean_valid)),
        "bottleneck_index_y_interior": float(1.0 - np.nanmin(mean_interior) / np.nanmedian(mean_interior)),
        "total_bottleneck_index_y_interior": float(1.0 - np.nanmin(total_interior) / np.nanmedian(total_interior)),
        "min_plane_y_index": min_plane_idx,
    }


def save_csv(rows: list[dict], out_dir: Path) -> None:
    path = out_dir / "representative30_flux_heterogeneity_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def save_metric_bars(rows: list[dict], out_dir: Path) -> None:
    metrics = [
        ("cv", "CV"),
        ("gini", "Gini"),
        ("top_10pct_flux_share", "top 10%\nshare"),
        ("localization_index_1_minus_participation", "1 -\nparticipation"),
        ("bottleneck_index_y_interior", "interior y\nbottleneck"),
    ]
    quantities = sorted({str(row["quantity"]) for row in rows})
    for quantity in quantities:
        quantity_label = QUANTITY_LABELS.get(quantity, quantity)
        qrows = [row for row in rows if row["quantity"] == quantity]
        x = np.arange(len(metrics), dtype=np.float32)
        width = 0.34
        fig, ax = plt.subplots(
            figsize=panel_figsize(1, 1, panel_width_mm=98.0, extra_width_mm=20.0),
            constrained_layout=True,
        )
        for i, row in enumerate(qrows):
            vals = [float(row[key]) for key, _ in metrics]
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=row["sample"])
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics], rotation=20, ha="right")
        ax.set_ylabel("Metric value")
        ax.set_title(f"30 um {quantity_label} heterogeneity")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
        style_axes(ax, x_major=False)
        path = out_dir / f"representative30_{quantity}_heterogeneity_metrics.png"
        save_figure(fig, path)
        plt.close(fig)
        print(f"Saved {path}")


def save_lorenz(entries: list[dict], out_dir: Path, quantity: str) -> None:
    quantity_label = QUANTITY_LABELS.get(quantity, quantity)
    fig, ax = plt.subplots(figsize=panel_figsize(1, 1), constrained_layout=True)
    for entry in entries:
        scalar = entry["quantities"][quantity]
        vals = scalar[np.isfinite(scalar) & entry["se_mask"]].astype(np.float64, copy=False)
        vals = np.sort(vals[vals >= 0])
        total = vals.sum()
        if total <= 0:
            continue
        cum_flux = np.concatenate([[0.0], np.cumsum(vals) / total])
        cum_voxels = np.linspace(0.0, 1.0, cum_flux.size)
        ax.plot(cum_voxels, cum_flux, lw=2.4, label=entry["sample"])
    ax.plot([0, 1], [0, 1], color="0.6", lw=1.2, ls="--", label="uniform")
    ax.set_xlabel("SE voxel fraction")
    ax.set_ylabel("Flux fraction")
    ax.set_title(f"30 um Lorenz: {quantity_label}")
    ax.legend(loc="upper left")
    style_axes(ax)
    path = out_dir / f"representative30_{quantity}_lorenz_curve.png"
    save_figure(fig, path)
    plt.close(fig)
    print(f"Saved {path}")


def save_y_profiles(entries: list[dict], out_dir: Path, quantity: str, voxel_um: float,
                    margin_fraction: float) -> None:
    quantity_label = QUANTITY_LABELS.get(quantity, quantity)
    fig, axes = plt.subplots(1, 2, figsize=panel_figsize(2, 1), constrained_layout=True)
    for entry in entries:
        profile = plane_profile(entry["quantities"][quantity], entry["se_mask"])
        y = np.arange(profile["mean"].size, dtype=np.float32) * voxel_um
        mean = profile["mean"]
        total = profile["total"]
        interior = interior_slice(mean.size, margin_fraction)
        axes[0].plot(y, mean / np.nanmedian(mean[interior]), lw=2.2, label=entry["sample"])
        axes[1].plot(y, total / np.nanmedian(total[interior]), lw=2.2, label=entry["sample"])
    for ax in axes:
        ax.axhline(1.0, color="0.5", lw=1.0, ls="--")
        ax.axvspan(0, margin_fraction * y[-1], color="0.8", alpha=0.2, lw=0)
        ax.axvspan((1.0 - margin_fraction) * y[-1], y[-1], color="0.8", alpha=0.2, lw=0)
        ax.legend(loc="upper right")
        style_axes(ax)
    axes[0].set_xlabel("y (um)")
    axes[0].set_ylabel("Mean / interior median")
    axes[0].set_title(f"Plane mean: {quantity_label}")
    axes[1].set_xlabel("y (um)")
    axes[1].set_ylabel("Total / interior median")
    axes[1].set_title(f"Plane total: {quantity_label}")
    path = out_dir / f"representative30_{quantity}_bottleneck_y_profiles.png"
    save_figure(fig, path)
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


def transparent_cmap(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad((0.92, 0.92, 0.92, 1.0))
    return cmap


def masked_center_slice(entry: dict, field: str, plane: str) -> np.ma.MaskedArray:
    volume = entry[field] if field in entry else entry["quantities"][field]
    image = center_slices_xyz(volume)[plane]
    se = center_slices_xyz(entry["se_mask"])[plane]
    return np.ma.array(image, mask=~se)


def se_values(entries: list[dict], field: str) -> np.ndarray:
    values = []
    for entry in entries:
        arr = entry[field]
        vals = arr[np.isfinite(arr) & entry["se_mask"]]
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


def save_potential_linear_flux_maps(
    entries: list[dict],
    out_dir: Path,
    voxel_um: float,
    potential_limits: tuple[float, float],
    flux_limits: tuple[float, float],
    flux_field: str,
    flux_title: str,
    flux_label: str,
    filename_token: str,
) -> None:
    potential_cmap = transparent_cmap("viridis")
    flux_cmap = transparent_cmap("magma")
    for plane in PLANES:
        fig, axes = plt.subplots(
            2,
            len(entries),
            figsize=panel_figsize(len(entries), 2, extra_width_mm=20.0),
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.08, right=0.86, bottom=0.10, top=0.91, wspace=0.32, hspace=0.52)
        if len(entries) == 1:
            axes = axes[:, None]
        last_images = {}
        for col, entry in enumerate(entries):
            subvolume_id = entry["meta"].get("subvolume_id", "representative")
            for row, (field, title, cmap, limits) in enumerate(
                (
                    ("potential", "Potential / concentration", potential_cmap, potential_limits),
                    (flux_field, flux_title, flux_cmap, flux_limits),
                )
            ):
                ax = axes[row, col]
                image = masked_center_slice(entry, field, plane)
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
                if row == 1:
                    ax.set_xlabel(f"{axis_h} (um)")
                else:
                    ax.tick_params(labelbottom=False)
                ax.set_ylabel(f"{axis_v} (um)")
                ax.set_title(f"{entry['sample']} {subvolume_id}\n{title}")
        for row, (field, label) in enumerate(
            (("potential", "Potential / concentration"), (flux_field, flux_label))
        ):
            cbar = fig.colorbar(last_images[field], ax=axes[row, :], shrink=0.80, label=label)
            style_colorbar(cbar)
        style_axes(axes)
        path = out_dir / f"representative30_potential_{filename_token}_linear_comparison_{plane}.png"
        save_figure(fig, path)
        plt.close(fig)
        print(f"Saved {path}")


def save_low_high_maps(entries: list[dict], out_dir: Path, quantity: str, voxel_um: float,
                       global_median: float) -> None:
    quantity_label = QUANTITY_LABELS.get(quantity, quantity)
    for plane in PLANES:
        fig, axes = plt.subplots(1, len(entries), figsize=panel_figsize(len(entries), 1, extra_width_mm=24.0),
                                 constrained_layout=False)
        fig.subplots_adjust(left=0.07, right=0.82, bottom=0.12, top=0.86, wspace=0.28)
        if len(entries) == 1:
            axes = [axes]
        for ax, entry in zip(axes, entries):
            scalar = center_slices_xyz(entry["quantities"][quantity])[plane]
            se = center_slices_xyz(entry["se_mask"])[plane]
            image = np.full((*scalar.shape, 4), (0.92, 0.92, 0.92, 1.0), dtype=np.float32)
            image[se] = (0.64, 0.64, 0.64, 1.0)
            low = se & (scalar < 0.5 * global_median)
            high = se & (scalar > 2.0 * global_median)
            top10_threshold = np.nanpercentile(entry["quantities"][quantity][entry["se_mask"]], 90.0)
            top10 = se & (scalar > top10_threshold)
            image[low] = matplotlib.colors.to_rgba("#4575b4", alpha=0.95)
            image[high] = matplotlib.colors.to_rgba("#f46d43", alpha=0.95)
            image[top10] = matplotlib.colors.to_rgba("#d73027", alpha=0.95)
            ax.imshow(
                image.transpose(1, 0, 2),
                origin="lower",
                interpolation="nearest",
                extent=image_extent_um(scalar, plane, voxel_um),
                aspect="equal",
            )
            axis_h, axis_v = PLANE_AXES[plane]
            ax.set_xlabel(f"{axis_h} (um)")
            ax.set_ylabel(f"{axis_v} (um)")
            ax.set_title(f"{entry['sample']} {PLANE_LABELS[plane]}\n{quantity_label} localization")
        handles = [
            plt.Line2D([0], [0], color="#4575b4", lw=8, label="< 0.5x global median"),
            plt.Line2D([0], [0], color=(0.64, 0.64, 0.64), lw=8, label="SE background"),
            plt.Line2D([0], [0], color="#f46d43", lw=8, label="> 2x global median"),
            plt.Line2D([0], [0], color="#d73027", lw=8, label="sample top 10%"),
            plt.Line2D([0], [0], color=(0.92, 0.92, 0.92), lw=8, label="non-SE"),
        ]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.84, 0.50), frameon=False)
        path = out_dir / f"representative30_{quantity}_low_high_map_{plane}.png"
        style_axes(axes)
        save_figure(fig, path)
        plt.close(fig)
        print(f"Saved {path}")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    representative_root = Path(args.representative_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        load_sample_fields(project_root, representative_root, sample, args.se_label, args.se_threshold, args.direction)
        for sample in SAMPLES
    ]
    potential_limits = percentile_limits(
        se_values(entries, "potential"), args.potential_p_low, args.potential_p_high
    )
    flux_values = np.concatenate(
        [
            entry["quantities"]["flux_magnitude"][entry["se_mask"]]
            for entry in entries
            if np.any(entry["se_mask"])
        ]
    )
    flux_limits = percentile_limits(flux_values, args.flux_p_low, args.flux_p_high)
    save_potential_linear_flux_maps(
        entries,
        out_dir,
        args.voxel_size_um,
        potential_limits,
        flux_limits,
        "flux_magnitude",
        "Flux magnitude",
        "Flux magnitude",
        "flux",
    )

    abs_jy_key = next((key for key in entries[0]["quantities"] if key.lower() == "abs_jy"), None)
    if abs_jy_key is not None and all(abs_jy_key in entry["quantities"] for entry in entries):
        jy_values = np.concatenate(
            [
                entry["quantities"][abs_jy_key][entry["se_mask"]]
                for entry in entries
                if np.any(entry["se_mask"])
            ]
        )
        jy_limits = percentile_limits(jy_values, args.flux_p_low, args.flux_p_high)
        save_potential_linear_flux_maps(
            entries,
            out_dir,
            args.voxel_size_um,
            potential_limits,
            jy_limits,
            abs_jy_key,
            "|Jy|",
            "|Jy|",
            "abs_Jy",
        )

    quantities = sorted(set().union(*(entry["quantities"].keys() for entry in entries)))
    rows = []
    for quantity in quantities:
        all_values = []
        for entry in entries:
            scalar = entry["quantities"][quantity]
            vals = scalar[np.isfinite(scalar) & entry["se_mask"]]
            vals = vals[vals >= 0]
            all_values.append(vals.astype(np.float32, copy=False))
        global_median = float(np.nanmedian(np.concatenate(all_values)))
        for entry in entries:
            rows.append(
                metrics_for_values(
                    entry["sample"],
                    quantity,
                    entry["quantities"][quantity],
                    entry["se_mask"],
                    global_median,
                    args.interior_margin_fraction,
                    entry["meta"],
                )
            )
        save_lorenz(entries, out_dir, quantity)
        save_y_profiles(entries, out_dir, quantity, args.voxel_size_um, args.interior_margin_fraction)
        save_low_high_maps(entries, out_dir, quantity, args.voxel_size_um, global_median)

    save_csv(rows, out_dir)
    save_metric_bars(rows, out_dir)

    print("\nKey representative 30 um heterogeneity metrics:")
    for row in rows:
        print(
            f"{row['sample']} {row['quantity']}: keff={row['keff_norm']}, "
            f"CV={row['cv']:.3f}, Gini={row['gini']:.3f}, "
            f"top10_share={row['top_10pct_flux_share']:.3f}, "
            f"participation={row['participation_ratio']:.3f}, "
            f"interior_y_bottleneck={row['bottleneck_index_y_interior']:.3f}"
        )

    del entries
    gc.collect()


if __name__ == "__main__":
    main()

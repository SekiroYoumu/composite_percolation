from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile as tiff


SAMPLES = ("WM", "PFDT")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPACITY = 0.55


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render whole-ROI 3D cutaway views with PyVista. "
            "The default view colors the SE phase by potential or flux magnitude and clips the block open."
        )
    )
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument("--whole-roi-root", default=PROJECT_ROOT / "server_results" / "whole_roi")
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "viz" / "whole_roi_3d_cutaway")
    parser.add_argument("--quantity", choices=["potential", "flux_magnitude"], default="potential")
    parser.add_argument(
        "--flux-display",
        choices=["raw", "relative", "log2_relative"],
        default="log2_relative",
        help=(
            "Display transform for flux_magnitude. log2_relative maps 0.5x/1x/2x "
            "global SE median flux to -1/0/+1, matching the heterogeneity analysis."
        ),
    )
    parser.add_argument(
        "--flux-log2-limit",
        type=float,
        default=1.0,
        help="Symmetric color limit for log2_relative flux display. The default highlights <0.5x and >2x median flux.",
    )
    parser.add_argument(
        "--highlight-high-flux",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay high-flux SE regions as opaque red channels when using log2_relative flux display.",
    )
    parser.add_argument(
        "--highlight-flux-mode",
        choices=["top_percentile", "median_2x"],
        default="top_percentile",
        help="High-flux overlay threshold: sample percentile cutoff, or flux > 2x global SE median.",
    )
    parser.add_argument(
        "--highlight-flux-percentile",
        type=float,
        default=95.0,
        help="Percentile cutoff for the high-flux overlay when --highlight-flux-mode=top_percentile.",
    )
    parser.add_argument("--se-label", type=int, default=2)
    parser.add_argument("--se-threshold", type=float, default=0.5)
    parser.add_argument("--voxel-size-um", type=float, default=0.07)
    parser.add_argument("--downsample-factor", type=int, default=None)
    parser.add_argument("--crop-fraction", type=float, default=1.0, help="Center crop fraction for faster draft rendering.")
    parser.add_argument("--opacity", type=float, default=DEFAULT_OPACITY)
    parser.add_argument("--window-width", type=int, default=2200)
    parser.add_argument("--window-height", type=int, default=1200)
    return parser.parse_args()


def require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise SystemExit(
            "PyVista is not installed in this local environment. Install it before 3D rendering, for example:\n"
            "  pip install pyvista vtk\n"
            "or\n"
            "  conda install -c conda-forge pyvista vtk\n"
        ) from exc
    return pv


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_fields(root: Path, sample: str) -> tuple[dict[str, np.ndarray], dict]:
    field_dir = root / "bulk_fields" / sample / "fields" / "whole_roi"
    npz_path = field_dir / "whole_roi_downsampled_fields.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    meta = read_json(field_dir / "metadata.json")
    with np.load(npz_path) as data:
        potential_key = "concentration" if "concentration" in data.files else "potential"
        fields = {
            "potential": np.asarray(data[potential_key], dtype=np.float32),
            "flux_magnitude": np.asarray(data["flux_magnitude"], dtype=np.float32),
        }
    return fields, meta


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


def center_crop(array: np.ndarray, fraction: float) -> np.ndarray:
    if fraction >= 0.999:
        return array
    slices = []
    for dim in array.shape:
        n = max(8, int(round(dim * fraction)))
        start = (dim - n) // 2
        slices.append(slice(start, start + n))
    return array[tuple(slices)]


def finite_se_values(scalar: np.ndarray, se_fraction: np.ndarray, se_threshold: float) -> np.ndarray:
    values = scalar[(se_fraction >= se_threshold) & np.isfinite(scalar)]
    return values.astype(np.float32, copy=False)


def display_scalar(raw_scalar: np.ndarray, quantity: str, flux_display: str,
                   global_flux_median: float | None) -> tuple[np.ndarray, str, str, tuple[float, float]]:
    if quantity != "flux_magnitude":
        return raw_scalar, "potential / concentration", "viridis", (1.0, 99.0)

    if flux_display == "raw":
        return raw_scalar, "flux magnitude", "inferno", (5.0, 99.0)

    if global_flux_median is None or not np.isfinite(global_flux_median) or global_flux_median <= 0:
        raise ValueError("Cannot make relative flux display because the global SE flux median is not positive.")

    relative = raw_scalar / global_flux_median
    if flux_display == "relative":
        return relative.astype(np.float32, copy=False), "flux / global SE median", "turbo", (5.0, 99.0)

    log_relative = np.full(raw_scalar.shape, np.nan, dtype=np.float32)
    positive = relative > 0
    log_relative[positive] = np.log2(relative[positive]).astype(np.float32, copy=False)
    return log_relative, "log2 flux / median", "coolwarm", (2.0, 98.0)


def make_grid(pv, scalar: np.ndarray, se_fraction: np.ndarray, spacing_um: float):
    common = tuple(min(scalar.shape[i], se_fraction.shape[i]) for i in range(3))
    slicer = tuple(slice(0, n) for n in common)
    scalar = scalar[slicer]
    se_fraction = se_fraction[slicer]
    grid = pv.ImageData()
    grid.dimensions = scalar.shape
    grid.spacing = (spacing_um, spacing_um, spacing_um)
    grid.point_data["value"] = scalar.ravel(order="F")
    grid.point_data["se_fraction"] = se_fraction.ravel(order="F")
    return grid


def add_cutaway_mesh(plotter, pv, grid, title: str, scalar_label: str, cmap: str, se_threshold: float,
                     clim: tuple[float, float], opacity: float, show_scalar_bar: bool,
                     high_flux_threshold: float | None = None) -> None:
    se_grid = grid.threshold(value=se_threshold, scalars="se_fraction")
    bounds = se_grid.bounds
    lx = bounds[1] - bounds[0]
    ly = bounds[3] - bounds[2]
    lz = bounds[5] - bounds[4]
    clip_bounds = (
        bounds[0] + 0.42 * lx,
        bounds[1],
        bounds[2] + 0.42 * ly,
        bounds[3],
        bounds[4],
        bounds[4] + 0.72 * lz,
    )
    cut = se_grid.clip_box(clip_bounds, invert=True)
    plotter.add_mesh(
        cut,
        scalars="value",
        cmap=cmap,
        clim=clim,
        opacity=opacity,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args={
            "title": scalar_label,
            "n_labels": 5,
            "fmt": "%.2f",
            "title_font_size": 12,
            "label_font_size": 11,
            "position_x": 0.88,
            "position_y": 0.25,
            "width": 0.03,
            "height": 0.48,
        } if show_scalar_bar else None,
        smooth_shading=False,
    )
    if high_flux_threshold is not None:
        high_flux = cut.threshold(value=high_flux_threshold, scalars="value")
        if high_flux.n_points:
            plotter.add_mesh(
                high_flux,
                color="#d73027",
                opacity=0.92,
                show_scalar_bar=False,
                smooth_shading=False,
            )
    outline = pv.Box(bounds=bounds).outline()
    plotter.add_mesh(outline, color="black", line_width=1.0)
    plotter.add_axes(line_width=2, labels_off=False)
    plotter.add_text(title, font_size=14)


def main() -> None:
    args = parse_args()
    pv = require_pyvista()
    pv.OFF_SCREEN = True

    project_root = Path(args.project_root)
    whole_roi_root = Path(args.whole_roi_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    raw_flux_for_median = []
    for sample in SAMPLES:
        fields, meta = load_fields(whole_roi_root, sample)
        downsample = args.downsample_factor or int(meta.get("downsample_factor", 2))
        scalar = center_crop(fields[args.quantity], args.crop_fraction)
        full_se = downsample_se_fraction(
            tiff.imread(label_path(project_root, sample)),
            args.se_label,
            downsample,
            fields[args.quantity].shape,
        )
        se_fraction = center_crop(full_se, args.crop_fraction)
        common = tuple(min(scalar.shape[i], se_fraction.shape[i]) for i in range(3))
        scalar = scalar[tuple(slice(0, n) for n in common)]
        se_fraction = se_fraction[tuple(slice(0, n) for n in common)]
        entries.append(
            {
                "sample": sample,
                "raw_scalar": scalar,
                "se_fraction": se_fraction,
                "downsample": downsample,
            }
        )
        vals = finite_se_values(scalar, se_fraction, args.se_threshold)
        if args.quantity == "flux_magnitude":
            vals = vals[vals > 0]
            if vals.size:
                raw_flux_for_median.append(vals)
        print(f"{sample}: scalar={scalar.shape}, se_fraction={se_fraction.shape}, downsample={downsample}")

    global_flux_median = None
    if args.quantity == "flux_magnitude":
        global_flux_median = float(np.nanmedian(np.concatenate(raw_flux_for_median)))
        print(f"Global SE flux median: {global_flux_median:.6g}")

    values_for_clim = []
    for entry in entries:
        scalar, scalar_label, cmap, percentiles = display_scalar(
            entry["raw_scalar"],
            args.quantity,
            args.flux_display,
            global_flux_median,
        )
        entry["scalar"] = scalar
        entry["scalar_label"] = scalar_label
        entry["cmap"] = cmap
        vals = finite_se_values(scalar, entry["se_fraction"], args.se_threshold)
        if vals.size:
            values_for_clim.append(vals)
        if (
            args.quantity == "flux_magnitude"
            and args.flux_display == "log2_relative"
            and args.highlight_high_flux
            and vals.size
        ):
            if args.highlight_flux_mode == "top_percentile":
                entry["highlight_threshold"] = float(np.nanpercentile(vals, args.highlight_flux_percentile))
            else:
                entry["highlight_threshold"] = float(args.flux_log2_limit)

    merged = np.concatenate(values_for_clim)
    clim = tuple(float(v) for v in np.nanpercentile(merged, percentiles))
    if args.quantity == "flux_magnitude" and args.flux_display == "log2_relative":
        limit = float(args.flux_log2_limit)
        clim = (-limit, limit)
    print(f"Shared color limits for {entries[0]['scalar_label']}: {clim}")

    plotter = pv.Plotter(shape=(1, len(entries)), off_screen=True, window_size=(args.window_width, args.window_height))
    for col, entry in enumerate(entries):
        plotter.subplot(0, col)
        spacing = args.voxel_size_um * entry["downsample"]
        opacity = args.opacity
        if args.quantity == "flux_magnitude" and args.opacity == DEFAULT_OPACITY:
            opacity = 0.46
        grid = make_grid(pv, entry["scalar"], entry["se_fraction"], spacing)
        add_cutaway_mesh(
            plotter,
            pv,
            grid,
            f"{entry['sample']} SE-only {entry['scalar_label']}",
            entry["scalar_label"],
            entry["cmap"],
            args.se_threshold,
            clim,
            opacity,
            show_scalar_bar=(col == len(entries) - 1),
            high_flux_threshold=entry.get("highlight_threshold"),
        )
        plotter.camera_position = "iso"
        plotter.camera.zoom(1.15)

    plotter.link_views()
    out_path = out_dir / f"whole_roi_3d_cutaway_se_{args.quantity}.png"
    plotter.screenshot(str(out_path))
    plotter.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

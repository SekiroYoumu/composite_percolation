from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile as tiff


SAMPLES = ("WM", "PFDT")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--se-label", type=int, default=2)
    parser.add_argument("--se-threshold", type=float, default=0.5)
    parser.add_argument("--voxel-size-um", type=float, default=0.07)
    parser.add_argument("--downsample-factor", type=int, default=None)
    parser.add_argument("--crop-fraction", type=float, default=1.0, help="Center crop fraction for faster draft rendering.")
    parser.add_argument("--opacity", type=float, default=0.55)
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


def add_cutaway_mesh(plotter, pv, grid, title: str, quantity: str, se_threshold: float,
                     clim: tuple[float, float], opacity: float) -> None:
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
    cut = se_grid.clip_box(clip_bounds, invert=False)
    plotter.add_mesh(
        cut,
        scalars="value",
        cmap="viridis" if quantity == "potential" else "magma",
        clim=clim,
        opacity=opacity,
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
    values_for_clim = []
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
        entries.append((sample, scalar, se_fraction, downsample))
        vals = scalar[se_fraction >= args.se_threshold]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            values_for_clim.append(vals.astype(np.float32, copy=False))
        print(f"{sample}: scalar={scalar.shape}, se_fraction={se_fraction.shape}, downsample={downsample}")

    merged = np.concatenate(values_for_clim)
    if args.quantity == "flux_magnitude":
        clim = tuple(float(v) for v in np.nanpercentile(merged, [5, 99]))
    else:
        clim = tuple(float(v) for v in np.nanpercentile(merged, [1, 99]))
    print(f"Shared color limits for {args.quantity}: {clim}")

    plotter = pv.Plotter(shape=(1, len(entries)), off_screen=True, window_size=(args.window_width, args.window_height))
    for col, (sample, scalar, se_fraction, downsample) in enumerate(entries):
        plotter.subplot(0, col)
        spacing = args.voxel_size_um * downsample
        grid = make_grid(pv, scalar, se_fraction, spacing)
        add_cutaway_mesh(plotter, pv, grid, f"{sample} SE-only {args.quantity}", args.quantity,
                         args.se_threshold, clim, args.opacity)
        plotter.camera_position = "iso"
        plotter.camera.zoom(1.15)

    plotter.link_views()
    plotter.add_scalar_bar(title=args.quantity, n_labels=5, position_x=0.88, position_y=0.25, height=0.5)
    out_path = out_dir / f"whole_roi_3d_cutaway_se_{args.quantity}.png"
    plotter.screenshot(str(out_path))
    plotter.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

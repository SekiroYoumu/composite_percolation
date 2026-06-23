from __future__ import annotations

import argparse
import json
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from matplotlib.colors import ListedColormap, to_rgb


SAMPLES = ("WM", "PFDT")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPACITY = 0.52
FLUX_QUANTITIES = {"flux_magnitude", "abs_Jy"}
SE_BASE_COLOR = "#b0b0b0"
LOW_FLUX_COLOR = "#4575b4"
HIGH_FLUX_COLOR = "#d73027"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    output_prefix: str
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render SE-only 3D cutaway views with PyVista for whole ROI and 30 um representative fields."
        )
    )
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument("--whole-roi-root", default=PROJECT_ROOT / "server_results" / "whole_roi")
    parser.add_argument(
        "--representative-root",
        default=PROJECT_ROOT / "server_results" / "30um-in-plane" / "results",
    )
    parser.add_argument(
        "--dataset",
        choices=["whole_roi", "representative30", "all"],
        default="all",
        help="Dataset to render. The default writes both whole ROI and 30 um representative cutaways.",
    )
    parser.add_argument(
        "--quantity",
        choices=["potential", "flux_magnitude", "abs_Jy", "all"],
        default="all",
        help="Scalar quantity to render. all expands to the quantities available for each dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output override. Use with a single --dataset. "
            "For --dataset all, use --whole-roi-output-dir and --representative30-output-dir."
        ),
    )
    parser.add_argument("--whole-roi-output-dir", default=PROJECT_ROOT / "viz" / "whole_roi_3d_cutaway")
    parser.add_argument(
        "--representative30-output-dir",
        default=PROJECT_ROOT / "viz" / "representative30_3d_cutaway",
    )
    parser.add_argument("--se-label", type=int, default=2)
    parser.add_argument("--am-label", type=int, default=1)
    parser.add_argument("--void-label", type=int, default=3)
    parser.add_argument("--se-threshold", type=float, default=0.5)
    parser.add_argument(
        "--potential-mask",
        choices=["se", "se_am"],
        default="se",
        help="Mask used for potential rendering. Flux quantities always use the SE-only mask.",
    )
    parser.add_argument("--voxel-size-um", type=float, default=0.07)
    parser.add_argument("--downsample-factor", type=int, default=None)
    parser.add_argument("--representative-downsample-factor", type=int, default=2)
    parser.add_argument("--crop-fraction", type=float, default=1.0, help="Center crop fraction for draft rendering.")
    parser.add_argument("--opacity", type=float, default=DEFAULT_OPACITY)
    parser.add_argument("--flux-relative-limit", type=float, default=2.0)
    parser.add_argument("--flux-tail-percent", type=float, default=5.0)
    parser.add_argument(
        "--flux-map-mode",
        choices=["tails", "high_only"],
        default="tails",
        help=(
            "Flux rendering mode. tails shows the low/high tails; high_only hides the low tail "
            "and emphasizes only high-flux bottleneck channels."
        ),
    )
    parser.add_argument(
        "--flux-tail-scale",
        choices=["sample_median", "absolute"],
        default="sample_median",
        help=(
            "Scale used before extracting low/high flux tails. sample_median emphasizes localization "
            "within each sample; absolute uses one raw-value threshold for WM and PFDT."
        ),
    )
    parser.add_argument(
        "--flux-tail-reference",
        choices=["merged", "wm", "pfdt"],
        default="merged",
        help="Reference distribution used to choose flux tail thresholds.",
    )
    parser.add_argument("--potential-percentile-low", type=float, default=1.0)
    parser.add_argument("--potential-percentile-high", type=float, default=99.0)
    parser.add_argument(
        "--overlay-potential-high-flux",
        action="store_true",
        help=(
            "Render a combined view: translucent potential background plus red SE-only high-flux "
            "regions. The selected --quantity must be a flux quantity."
        ),
    )
    parser.add_argument(
        "--overlay-potential-mask",
        choices=["se", "se_am", "both"],
        default="se_am",
        help="Potential background mask for overlay rendering. both writes SE-only and SE+AM versions.",
    )
    parser.add_argument("--overlay-potential-opacity", type=float, default=0.50)
    parser.add_argument(
        "--overlay-potential-color-strength",
        type=float,
        default=0.3,
        help="Blend strength for potential color tint over the gray SE background in overlay renders.",
    )
    parser.add_argument("--overlay-high-flux-opacity", type=float, default=0.6)
    parser.add_argument(
        "--overlay-flux-style",
        choices=["high_only", "gradient"],
        default="high_only",
        help="Overlay bottleneck style. gradient uses a nonlinear WM-reference continuous colormap.",
    )
    parser.add_argument(
        "--overlay-gradient-gamma",
        type=float,
        default=1.8,
        help="Nonlinear gamma for gradient bottleneck overlay. Values >1 suppress mid values.",
    )
    parser.add_argument(
        "--overlay-gradient-min-fraction",
        type=float,
        default=0.18,
        help="Only render flux values above this fraction of the high reference threshold in gradient mode.",
    )
    parser.add_argument(
        "--overlay-show-colorbar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show the muted potential colorbar in overlay renders.",
    )
    parser.add_argument(
        "--render-void-empty",
        action="store_true",
        help=(
            "Render the 30 um non-void matrix as a translucent solid and leave void/carbon-rich "
            "phase empty, emphasizing internal pore distribution."
        ),
    )
    parser.add_argument("--matrix-opacity", type=float, default=0.34)
    parser.add_argument("--matrix-color", default="#b8bec4")
    parser.add_argument("--void-cut-face-color", default="#f0b24d")
    parser.add_argument("--void-cut-face-opacity", type=float, default=0.92)
    parser.add_argument(
        "--no-void-cut-faces",
        action="store_true",
        help="Disable orange void/carbon-rich patches on the three exposed cut faces.",
    )
    parser.add_argument(
        "--no-void-empty-cutaway",
        action="store_true",
        help="Render the full transparent matrix cube instead of clipping a corner open.",
    )
    parser.add_argument("--highlight-high-flux", action=argparse.BooleanOptionalAction, default=True)
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


def center_crop(array: np.ndarray, fraction: float) -> np.ndarray:
    if fraction >= 0.999:
        return array
    slices = []
    for dim in array.shape:
        n = max(8, int(round(dim * fraction)))
        start = (dim - n) // 2
        slices.append(slice(start, start + n))
    return array[tuple(slices)]


def block_mean(array: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return array.astype(np.float32, copy=False)
    crop_shape = tuple((dim // factor) * factor for dim in array.shape)
    cropped = array[tuple(slice(0, n) for n in crop_shape)].astype(np.float32, copy=False)
    reshaped = cropped.reshape(
        crop_shape[0] // factor,
        factor,
        crop_shape[1] // factor,
        factor,
        crop_shape[2] // factor,
        factor,
    )
    return reshaped.mean(axis=(1, 3, 5), dtype=np.float32)


def downsample_label_fraction(
    label_zyx: np.ndarray,
    mask_labels: tuple[int, ...],
    factor: int,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    label_xyz = np.transpose(label_zyx, (2, 1, 0))
    mask = np.isin(label_xyz, mask_labels)
    if factor <= 1:
        common = tuple(min(a, b) for a, b in zip(mask.shape, target_shape))
        return mask[tuple(slice(0, n) for n in common)].astype(np.float32, copy=False)
    crop_shape = tuple(min(mask.shape[i], target_shape[i] * factor) for i in range(3))
    crop_shape = tuple((n // factor) * factor for n in crop_shape)
    cropped = mask[tuple(slice(0, n) for n in crop_shape)].astype(np.float32, copy=False)
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


def mask_labels_for_quantity(args: argparse.Namespace, quantity: str) -> tuple[int, ...]:
    if quantity == "potential" and args.potential_mask == "se_am":
        return (int(args.am_label), int(args.se_label))
    return (int(args.se_label),)


def mask_label_for_display(args: argparse.Namespace, quantity: str) -> str:
    if quantity == "potential" and args.potential_mask == "se_am":
        return "SE+AM"
    return "SE-only"


def finite_se_values(scalar: np.ndarray, se_fraction: np.ndarray, se_threshold: float) -> np.ndarray:
    values = scalar[(se_fraction >= se_threshold) & np.isfinite(scalar)]
    return values.astype(np.float32, copy=False)


def load_whole_roi_entry(args: argparse.Namespace, sample: str, quantity: str) -> dict:
    field_dir = Path(args.whole_roi_root) / "bulk_fields" / sample / "fields" / "whole_roi"
    npz_path = field_dir / "whole_roi_downsampled_fields.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    meta = read_json(field_dir / "metadata.json")
    with np.load(npz_path) as data:
        potential_key = "concentration" if "concentration" in data.files else "potential"
        key = potential_key if quantity == "potential" else quantity
        scalar = np.asarray(data[key], dtype=np.float32)

    downsample = args.downsample_factor or int(meta.get("downsample_factor", 2))
    se_fraction = downsample_label_fraction(
        tiff.imread(label_path(Path(args.project_root), sample)),
        mask_labels_for_quantity(args, quantity),
        downsample,
        scalar.shape,
    )
    scalar, se_fraction = common_crop(scalar, se_fraction)
    scalar = center_crop(scalar, args.crop_fraction)
    se_fraction = center_crop(se_fraction, args.crop_fraction)
    scalar, se_fraction = common_crop(scalar, se_fraction)
    return {
        "sample": sample,
        "scalar": scalar,
        "se_fraction": se_fraction,
        "spacing_um": float(args.voxel_size_um) * downsample,
        "downsample": downsample,
    }


def load_representative30_entry(args: argparse.Namespace, sample: str, quantity: str) -> dict:
    field_dir = Path(args.representative_root) / sample / "representative_flux"
    npz_path = field_dir / "median_representative_fields.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    meta = read_json(field_dir / "metadata.json")
    with np.load(npz_path) as data:
        if quantity == "potential":
            potential_key = "concentration" if "concentration" in data.files else "potential"
            scalar = np.asarray(data[potential_key], dtype=np.float32)
        elif quantity == "flux_magnitude":
            scalar = np.asarray(data["flux_magnitude"], dtype=np.float32)
        else:
            scalar = np.abs(np.asarray(data["flux"][..., 1], dtype=np.float32))

    label_zyx = tiff.imread(label_path(Path(args.project_root), sample))
    label_xyz = np.transpose(crop_label_to_metadata(label_zyx, meta), (2, 1, 0))
    se_fraction = np.isin(label_xyz, mask_labels_for_quantity(args, quantity)).astype(np.float32, copy=False)
    scalar, se_fraction = common_crop(scalar, se_fraction)
    scalar = center_crop(scalar, args.crop_fraction)
    se_fraction = center_crop(se_fraction, args.crop_fraction)
    scalar, se_fraction = common_crop(scalar, se_fraction)

    factor = int(args.downsample_factor or args.representative_downsample_factor)
    scalar = block_mean(scalar, factor)
    se_fraction = block_mean(se_fraction, factor)
    scalar, se_fraction = common_crop(scalar, se_fraction)
    return {
        "sample": sample,
        "scalar": scalar,
        "se_fraction": se_fraction,
        "spacing_um": float(args.voxel_size_um) * factor,
        "downsample": factor,
    }


def load_representative30_matrix_entry(args: argparse.Namespace, sample: str) -> dict:
    field_dir = Path(args.representative_root) / sample / "representative_flux"
    meta = read_json(field_dir / "metadata.json")
    label_zyx = tiff.imread(label_path(Path(args.project_root), sample))
    label_xyz = np.transpose(crop_label_to_metadata(label_zyx, meta), (2, 1, 0))
    matrix_fraction = np.isin(label_xyz, (int(args.am_label), int(args.se_label))).astype(np.float32, copy=False)
    void_fraction = (label_xyz == int(args.void_label)).astype(np.float32, copy=False)
    matrix_fraction = center_crop(matrix_fraction, args.crop_fraction)
    void_fraction = center_crop(void_fraction, args.crop_fraction)
    matrix_fraction, void_fraction = common_crop(matrix_fraction, void_fraction)

    factor = int(args.downsample_factor or args.representative_downsample_factor)
    matrix_fraction = block_mean(matrix_fraction, factor)
    void_fraction = block_mean(void_fraction, factor)
    matrix_fraction, void_fraction = common_crop(matrix_fraction, void_fraction)
    scalar = np.zeros_like(matrix_fraction, dtype=np.float32)
    return {
        "sample": sample,
        "scalar": scalar,
        "se_fraction": matrix_fraction,
        "void_fraction": void_fraction,
        "spacing_um": float(args.voxel_size_um) * factor,
        "downsample": factor,
    }


def display_entries(entries: list[dict], quantity: str, args: argparse.Namespace) -> dict:
    if quantity == "potential":
        values = np.concatenate(
            [finite_se_values(entry["scalar"], entry["se_fraction"], args.se_threshold) for entry in entries]
        )
        clim = tuple(float(v) for v in np.nanpercentile(values, [args.potential_percentile_low,
                                                                 args.potential_percentile_high]))
        return {"mode": "continuous", "label": "potential", "cmap": "viridis", "clim": clim}

    scaled_values = {}
    for entry in entries:
        values = finite_se_values(entry["scalar"], entry["se_fraction"], args.se_threshold)
        values = values[np.isfinite(values) & (values > 0)]
        if not values.size:
            continue
        if args.flux_tail_scale == "sample_median":
            sample_median = float(np.nanmedian(values))
            if sample_median <= 0 or not np.isfinite(sample_median):
                raise ValueError(f"Cannot normalize {entry['sample']} {quantity}; SE median is not positive.")
            entry["scalar"] = (entry["scalar"] / sample_median).astype(np.float32, copy=False)
            scaled_values[entry["sample"]] = values / sample_median
            entry["tail_scale_label"] = f"{entry['sample']} median={sample_median:.6g}"
        else:
            scaled_values[entry["sample"]] = values
            entry["tail_scale_label"] = "absolute"
    merged = np.concatenate(list(scaled_values.values()))
    reference_name = args.flux_tail_reference.upper() if args.flux_tail_reference != "merged" else "merged"
    reference_values = merged if args.flux_tail_reference == "merged" else scaled_values[reference_name]
    low_percent = float(args.flux_tail_percent)
    high_percent = 100.0 - low_percent
    low_threshold, high_threshold = (float(v) for v in np.nanpercentile(reference_values, [low_percent, high_percent]))
    median = float(np.nanmedian(merged))
    label = "abs(J)" if quantity == "flux_magnitude" else "abs(Jy)"
    scale_label = "sample-median scaled" if args.flux_tail_scale == "sample_median" else "absolute"
    print(
        f"Global SE {label} ({scale_label}, {reference_name} threshold): "
        f"median={median:.6g}, top{low_percent:g}%>={high_threshold:.6g}"
    )
    if args.flux_map_mode == "tails":
        print(f"  bottom{low_percent:g}%<={low_threshold:.6g}")
    for entry in entries:
        sample_values = finite_se_values(entry["scalar"], entry["se_fraction"], args.se_threshold)
        sample_values = sample_values[np.isfinite(sample_values)]
        if sample_values.size:
            low_fraction = float(np.count_nonzero(sample_values <= low_threshold) / sample_values.size)
            high_fraction = float(np.count_nonzero(sample_values >= high_threshold) / sample_values.size)
            if args.flux_map_mode == "tails":
                print(
                    f"  {entry['sample']} tails: "
                    f"bottom{low_percent:g}% global={100 * low_fraction:.2f}%, "
                    f"top{low_percent:g}% global={100 * high_fraction:.2f}%"
                )
            else:
                print(f"  {entry['sample']} high tail: top{low_percent:g}% global={100 * high_fraction:.2f}%")
    return {
        "mode": args.flux_map_mode,
        "label": label,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "tail_percent": low_percent,
        "scale_label": scale_label,
        "threshold_reference": args.flux_tail_reference,
    }


def make_grid(pv, scalar: np.ndarray, se_fraction: np.ndarray, spacing_um: float):
    scalar, se_fraction = common_crop(scalar, se_fraction)
    grid = pv.ImageData()
    grid.dimensions = scalar.shape
    grid.spacing = (spacing_um, spacing_um, spacing_um)
    grid.point_data["value"] = scalar.ravel(order="F")
    grid.point_data["se_fraction"] = se_fraction.ravel(order="F")
    return grid


def cutaway_grid(pv, grid, se_threshold: float):
    se_grid = grid.threshold(value=se_threshold, scalars="se_fraction")
    bounds = se_grid.bounds
    clip_bounds = cutaway_clip_bounds(bounds)
    cut = se_grid.clip_box(clip_bounds, invert=True)
    return cut, bounds


def cutaway_clip_bounds(bounds: tuple[float, float, float, float, float, float]):
    lx = bounds[1] - bounds[0]
    ly = bounds[3] - bounds[2]
    lz = bounds[5] - bounds[4]
    return (
        bounds[0] + 0.42 * lx,
        bounds[1],
        bounds[2] + 0.42 * ly,
        bounds[3],
        bounds[4] + 0.42 * lz,
        bounds[5],
    )


def add_continuous_cutaway_mesh(
    plotter,
    pv,
    grid,
    title: str,
    scalar_label: str,
    cmap: str,
    se_threshold: float,
    clim: tuple[float, float],
    opacity: float,
    show_scalar_bar: bool,
) -> tuple[float, float, float, float, float, float]:
    cut, bounds = cutaway_grid(pv, grid, se_threshold)
    plotter.add_mesh(
        cut,
        scalars="value",
        cmap=cmap,
        clim=clim,
        opacity=opacity,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args={
            "title": scalar_label,
            "vertical": True,
            "n_labels": 3,
            "fmt": "%.2f",
            "title_font_size": 10,
            "label_font_size": 9,
            "position_x": 0.865,
            "position_y": 0.22,
            "width": 0.06,
            "height": 0.42,
        } if show_scalar_bar else None,
        smooth_shading=False,
    )
    outline = pv.Box(bounds=bounds).outline()
    plotter.add_mesh(outline, color="black", line_width=0.9)
    plotter.add_axes(line_width=1, labels_off=False)
    plotter.add_text(title, position=(0.03, 0.93), font_size=10, viewport=True)
    return bounds


def add_void_empty_matrix_mesh(
    plotter,
    pv,
    grid,
    title: str,
    matrix_threshold: float,
    matrix_color: str,
    matrix_opacity: float,
    void_cut_face_color: str,
    void_cut_face_opacity: float,
    show_void_cut_faces: bool,
    use_cutaway: bool,
) -> tuple[float, float, float, float, float, float]:
    matrix = grid.threshold(value=matrix_threshold, scalars="se_fraction")
    bounds = matrix.bounds
    if use_cutaway:
        clip_bounds = cutaway_clip_bounds(bounds)
        matrix = matrix.clip_box(clip_bounds, invert=True)
    else:
        clip_bounds = None
    plotter.add_mesh(
        matrix,
        color=matrix_color,
        opacity=matrix_opacity,
        show_scalar_bar=False,
        smooth_shading=False,
        lighting=True,
        ambient=0.38,
        diffuse=0.72,
        specular=0.08,
    )
    if show_void_cut_faces and clip_bounds is not None:
        add_void_cut_face_patches(
            plotter,
            pv,
            grid,
            bounds,
            clip_bounds,
            void_cut_face_color,
            void_cut_face_opacity,
        )
    outline = pv.Box(bounds=bounds).outline()
    plotter.add_mesh(outline, color="black", line_width=0.9)
    plotter.add_axes(line_width=1, labels_off=False)
    plotter.add_text(title, position=(0.03, 0.93), font_size=10, viewport=True)
    return bounds


def add_void_cut_face_patches(
    plotter,
    pv,
    grid,
    bounds: tuple[float, float, float, float, float, float],
    clip_bounds: tuple[float, float, float, float, float, float],
    color: str,
    opacity: float,
    threshold: float = 0.5,
) -> None:
    eps = min(grid.spacing) * 0.35
    faces = [
        (
            (1, 0, 0),
            (clip_bounds[0] + eps, 0.5 * (clip_bounds[2] + bounds[3]), 0.5 * (clip_bounds[4] + bounds[5])),
            (clip_bounds[0] - eps, clip_bounds[0] + 2 * eps, clip_bounds[2], bounds[3], clip_bounds[4], bounds[5]),
        ),
        (
            (0, 1, 0),
            (0.5 * (clip_bounds[0] + bounds[1]), clip_bounds[2] + eps, 0.5 * (clip_bounds[4] + bounds[5])),
            (clip_bounds[0], bounds[1], clip_bounds[2] - eps, clip_bounds[2] + 2 * eps, clip_bounds[4], bounds[5]),
        ),
        (
            (0, 0, 1),
            (0.5 * (clip_bounds[0] + bounds[1]), 0.5 * (clip_bounds[2] + bounds[3]), clip_bounds[4] + eps),
            (clip_bounds[0], bounds[1], clip_bounds[2], bounds[3], clip_bounds[4] - eps, clip_bounds[4] + 2 * eps),
        ),
    ]
    for normal, origin, face_bounds in faces:
        face = grid.slice(normal=normal, origin=origin)
        if not face.n_points:
            continue
        void_face = face.threshold(value=threshold, scalars="void_fraction")
        if not void_face.n_points:
            continue
        void_face = void_face.clip_box(face_bounds, invert=False)
        if not void_face.n_points:
            continue
        plotter.add_mesh(
            void_face,
            color=color,
            opacity=opacity,
            show_scalar_bar=False,
            smooth_shading=False,
            lighting=False,
        )


def add_tail_cutaway_mesh(
    plotter,
    pv,
    grid,
    title: str,
    se_threshold: float,
    low_threshold: float,
    high_threshold: float,
    tail_percent: float,
    show_legend: bool,
) -> tuple[float, float, float, float, float, float]:
    cut, bounds = cutaway_grid(pv, grid, se_threshold)
    plotter.add_mesh(
        cut,
        color=SE_BASE_COLOR,
        opacity=0.90,
        show_scalar_bar=False,
        smooth_shading=False,
        lighting=False,
    )
    low_flux = cut.threshold(value=low_threshold, scalars="value", method="lower", all_scalars=True)
    if low_flux.n_points:
        plotter.add_mesh(
            low_flux,
            color=LOW_FLUX_COLOR,
            opacity=0.75,
            show_scalar_bar=False,
            smooth_shading=False,
            lighting=False,
        )
    high_flux = cut.threshold(value=high_threshold, scalars="value", method="upper", all_scalars=True)
    if high_flux.n_points:
        plotter.add_mesh(
            high_flux,
            color=HIGH_FLUX_COLOR,
            opacity=0.75,
            show_scalar_bar=False,
            smooth_shading=False,
            lighting=False,
        )
    outline = pv.Box(bounds=bounds).outline()
    plotter.add_mesh(outline, color="black", line_width=0.9)
    plotter.add_axes(line_width=1, labels_off=False)
    plotter.add_text(title, position=(0.03, 0.93), font_size=10, viewport=True)
    if show_legend:
        plotter.add_legend(
            labels=[
                [f"middle {100 - 2 * tail_percent:.0f}%", SE_BASE_COLOR],
                [f"low {tail_percent:.0f}%", LOW_FLUX_COLOR],
                [f"high {tail_percent:.0f}%", HIGH_FLUX_COLOR],
            ],
            size=(0.18, 0.13),
            loc="lower right",
            bcolor="white",
            border=False,
            background_opacity=0.0,
        )
    return bounds


def add_high_only_cutaway_mesh(
    plotter,
    pv,
    grid,
    title: str,
    se_threshold: float,
    high_threshold: float,
    tail_percent: float,
    show_legend: bool,
) -> tuple[float, float, float, float, float, float]:
    cut, bounds = cutaway_grid(pv, grid, se_threshold)
    plotter.add_mesh(
        cut,
        color=SE_BASE_COLOR,
        opacity=0.50,
        show_scalar_bar=False,
        smooth_shading=False,
        lighting=False,
    )
    high_flux = cut.threshold(value=high_threshold, scalars="value", method="upper", all_scalars=True)
    if high_flux.n_points:
        plotter.add_mesh(
            high_flux,
            color=HIGH_FLUX_COLOR,
            opacity=0.98,
            show_scalar_bar=False,
            smooth_shading=False,
            lighting=False,
        )
    outline = pv.Box(bounds=bounds).outline()
    plotter.add_mesh(outline, color="black", line_width=0.9)
    plotter.add_axes(line_width=1, labels_off=False)
    plotter.add_text(title, position=(0.03, 0.93), font_size=10, viewport=True)
    if show_legend:
        plotter.add_legend(
            labels=[
                ["SE", SE_BASE_COLOR],
                [f"top {tail_percent:.0f}%", HIGH_FLUX_COLOR],
            ],
            size=(0.12, 0.08),
            loc="lower right",
            bcolor="white",
            border=False,
            background_opacity=0.0,
        )
    return bounds


def muted_potential_cmap(color_strength: float) -> ListedColormap:
    strength = float(np.clip(color_strength, 0.0, 1.0))
    base = np.array(to_rgb(SE_BASE_COLOR), dtype=np.float32)
    viridis = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, 256))[:, :3].astype(np.float32)
    colors = (1.0 - strength) * base[None, :] + strength * viridis
    return ListedColormap(colors, name=f"gray_viridis_{strength:.2f}")


def bottleneck_gradient_cmap(gamma: float) -> ListedColormap:
    gamma = max(float(gamma), 1.0e-3)
    base = np.array(
        [
            to_rgb("#6f6f83"),
            to_rgb("#7f6ea8"),
            to_rgb("#d6604d"),
            to_rgb("#fff176"),
        ],
        dtype=np.float32,
    )
    anchors = np.linspace(0.0, 1.0, base.shape[0])
    x = np.linspace(0.0, 1.0, 256) ** gamma
    colors = np.empty((256, 3), dtype=np.float32)
    for channel in range(3):
        colors[:, channel] = np.interp(x, anchors, base[:, channel])
    return ListedColormap(colors, name=f"bottleneck_gamma_{gamma:.2f}")


def add_potential_high_flux_overlay_mesh(
    plotter,
    pv,
    potential_grid,
    flux_grid,
    title: str,
    se_threshold: float,
    potential_clim: tuple[float, float],
    high_threshold: float,
    tail_percent: float,
    potential_opacity: float,
    high_flux_opacity: float,
    potential_color_strength: float,
    flux_style: str,
    gradient_gamma: float,
    gradient_min_fraction: float,
    show_scalar_bar: bool,
    show_legend: bool,
) -> tuple[float, float, float, float, float, float]:
    potential_cut, bounds = cutaway_grid(pv, potential_grid, se_threshold)
    plotter.add_mesh(
        potential_cut,
        scalars="value",
        cmap=muted_potential_cmap(potential_color_strength),
        clim=potential_clim,
        opacity=potential_opacity,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args={
            "title": "potential",
            "vertical": True,
            "n_labels": 3,
            "fmt": "%.2f",
            "title_font_size": 10,
            "label_font_size": 9,
            "position_x": 0.865,
            "position_y": 0.24,
            "width": 0.055,
            "height": 0.38,
        } if show_scalar_bar else None,
        smooth_shading=False,
        lighting=False,
    )
    flux_cut, _ = cutaway_grid(pv, flux_grid, se_threshold)
    if flux_style == "gradient":
        gradient_floor = max(0.0, float(gradient_min_fraction)) * high_threshold
        highlighted_flux = flux_cut.threshold(value=gradient_floor, scalars="value", method="upper", all_scalars=True)
        if highlighted_flux.n_points:
            plotter.add_mesh(
                highlighted_flux,
                scalars="value",
                cmap=bottleneck_gradient_cmap(gradient_gamma),
                clim=(0.0, high_threshold),
                opacity=high_flux_opacity,
                show_scalar_bar=False,
                smooth_shading=False,
                lighting=False,
            )
    else:
        high_flux = flux_cut.threshold(value=high_threshold, scalars="value", method="upper", all_scalars=True)
        if high_flux.n_points:
            plotter.add_mesh(
                high_flux,
                color=HIGH_FLUX_COLOR,
                opacity=high_flux_opacity,
                show_scalar_bar=False,
                smooth_shading=False,
                lighting=False,
            )
    outline = pv.Box(bounds=bounds).outline()
    plotter.add_mesh(outline, color="black", line_width=0.9)
    plotter.add_axes(line_width=1, labels_off=False)
    plotter.add_text(title, position=(0.03, 0.93), font_size=10, viewport=True)
    if show_legend:
        if flux_style == "gradient":
            labels = [
                ["potential", SE_BASE_COLOR],
                ["high flux", "#d6604d"],
                [f">= ref top {tail_percent:.0f}%", "#fff176"],
            ]
        else:
            labels = [
                ["potential", SE_BASE_COLOR],
                [f"top {tail_percent:.0f}%", HIGH_FLUX_COLOR],
            ]
        plotter.add_legend(
            labels=labels,
            size=(0.12, 0.08),
            loc="lower right",
            bcolor="white",
            border=False,
            background_opacity=0.0,
        )
    return bounds


def set_y_up_camera(plotter, bounds: tuple[float, float, float, float, float, float]) -> None:
    center = np.array(
        [
            0.5 * (bounds[0] + bounds[1]),
            0.5 * (bounds[2] + bounds[3]),
            0.5 * (bounds[4] + bounds[5]),
        ],
        dtype=np.float64,
    )
    span = np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]], dtype=np.float64)
    distance = float(np.linalg.norm(span) * 2.25)
    direction = np.array([1.0, 0.78, 1.0], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    plotter.camera.position = tuple(center + distance * direction)
    plotter.camera.focal_point = tuple(center)
    plotter.camera.view_up = (0.0, 1.0, 0.0)
    plotter.camera.zoom(0.98)


def dataset_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    whole_dir = Path(args.output_dir) if args.output_dir and args.dataset == "whole_roi" else Path(args.whole_roi_output_dir)
    rep_dir = (
        Path(args.output_dir)
        if args.output_dir and args.dataset == "representative30"
        else Path(args.representative30_output_dir)
    )
    specs = {
        "whole_roi": DatasetSpec("whole_roi", "whole_roi_3d_cutaway", whole_dir),
        "representative30": DatasetSpec("representative30", "representative30_3d_cutaway", rep_dir),
    }
    if args.dataset == "all":
        return [specs["whole_roi"], specs["representative30"]]
    return [specs[args.dataset]]


def quantities_for_dataset(dataset: str, requested: str) -> list[str]:
    available = {
        "whole_roi": ["potential", "flux_magnitude"],
        "representative30": ["potential", "flux_magnitude", "abs_Jy"],
    }[dataset]
    if requested == "all":
        return available
    if requested not in available:
        raise ValueError(f"{requested} is not available for {dataset}. Available quantities: {', '.join(available)}")
    return [requested]


def load_entries(args: argparse.Namespace, dataset: str, quantity: str) -> list[dict]:
    loader = load_whole_roi_entry if dataset == "whole_roi" else load_representative30_entry
    mask_label = mask_label_for_display(args, quantity)
    entries = [loader(args, sample, quantity) for sample in SAMPLES]
    for entry in entries:
        print(
            f"{dataset} {entry['sample']} {mask_label} {quantity}: "
            f"scalar={entry['scalar'].shape}, mask_fraction={entry['se_fraction'].shape}, "
            f"downsample={entry['downsample']}, spacing={entry['spacing_um']:.3g} um"
        )
    return entries


def render_cutaway(args: argparse.Namespace, pv, spec: DatasetSpec, quantity: str) -> Path:
    entries = load_entries(args, spec.name, quantity)
    display = display_entries(entries, quantity, args)
    mask_label = mask_label_for_display(args, quantity)
    if display["mode"] == "continuous":
        print(f"Shared color limits for {spec.name} {display['label']}: {display['clim']}")

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(shape=(1, len(entries)), off_screen=True, window_size=(args.window_width, args.window_height))
    plotter.set_background("white")
    first_bounds = None
    for col, entry in enumerate(entries):
        plotter.subplot(0, col)
        opacity = args.opacity
        if quantity in FLUX_QUANTITIES and args.opacity == DEFAULT_OPACITY:
            opacity = 0.38
        grid = make_grid(pv, entry["scalar"], entry["se_fraction"], entry["spacing_um"])
        if display["mode"] == "continuous":
            bounds = add_continuous_cutaway_mesh(
                plotter,
                pv,
                grid,
                f"{entry['sample']} {mask_label} {display['label']}",
                display["label"],
                display["cmap"],
                args.se_threshold,
                display["clim"],
                opacity,
                show_scalar_bar=(col == len(entries) - 1),
            )
        elif display["mode"] == "tails":
            bounds = add_tail_cutaway_mesh(
                plotter,
                pv,
                grid,
                f"{entry['sample']} SE-only {display['label']} tails",
                args.se_threshold,
                display["low_threshold"],
                display["high_threshold"],
                display["tail_percent"],
                show_legend=(col == len(entries) - 1),
            )
        else:
            bounds = add_high_only_cutaway_mesh(
                plotter,
                pv,
                grid,
                f"{entry['sample']} SE-only {display['label']} high flux",
                args.se_threshold,
                display["high_threshold"],
                display["tail_percent"],
                show_legend=(col == len(entries) - 1),
            )
        set_y_up_camera(plotter, bounds)
        if first_bounds is None:
            first_bounds = bounds

    plotter.link_views()
    if first_bounds is not None:
        set_y_up_camera(plotter, first_bounds)
    mask_suffix = "_se_am" if quantity == "potential" and args.potential_mask == "se_am" else "_se"
    reference_suffix = ""
    if quantity in FLUX_QUANTITIES and display.get("threshold_reference") != "merged":
        reference_suffix = f"_{display['threshold_reference']}_ref"
    mode_suffix = "_high_only" if display["mode"] == "high_only" else ""
    out_path = spec.output_dir / f"{spec.output_prefix}{mask_suffix}_{quantity}{reference_suffix}{mode_suffix}.png"
    plotter.screenshot(str(out_path), transparent_background=True)
    plotter.close()
    print(f"Saved {out_path}")
    return out_path


def render_potential_high_flux_overlay(
    args: argparse.Namespace, pv, spec: DatasetSpec, flux_quantity: str, potential_mask: str
) -> Path:
    if flux_quantity not in FLUX_QUANTITIES:
        raise ValueError("--overlay-potential-high-flux requires --quantity flux_magnitude or abs_Jy.")
    if spec.name == "whole_roi" and flux_quantity == "abs_Jy":
        raise ValueError("whole_roi does not contain flux vectors, so abs_Jy overlay is unavailable.")

    potential_args = copy(args)
    potential_args.potential_mask = potential_mask
    potential_entries = load_entries(potential_args, spec.name, "potential")
    potential_display = display_entries(potential_entries, "potential", potential_args)
    print(f"Shared color limits for {spec.name} overlay potential: {potential_display['clim']}")

    flux_entries = load_entries(args, spec.name, flux_quantity)
    flux_display = display_entries(flux_entries, flux_quantity, args)
    potential_label = "SE+AM potential" if potential_mask == "se_am" else "SE-only potential"
    mask_label = f"{potential_label} + SE-only"

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(shape=(1, len(flux_entries)), off_screen=True, window_size=(args.window_width, args.window_height))
    plotter.set_background("white")
    first_bounds = None
    for col, (potential_entry, flux_entry) in enumerate(zip(potential_entries, flux_entries)):
        plotter.subplot(0, col)
        potential_grid = make_grid(
            pv,
            potential_entry["scalar"],
            potential_entry["se_fraction"],
            potential_entry["spacing_um"],
        )
        flux_grid = make_grid(pv, flux_entry["scalar"], flux_entry["se_fraction"], flux_entry["spacing_um"])
        bounds = add_potential_high_flux_overlay_mesh(
            plotter,
            pv,
            potential_grid,
            flux_grid,
            f"{flux_entry['sample']} {mask_label} {flux_display['label']} high flux",
            args.se_threshold,
            potential_display["clim"],
            flux_display["high_threshold"],
            flux_display["tail_percent"],
            args.overlay_potential_opacity,
            args.overlay_high_flux_opacity,
            args.overlay_potential_color_strength,
            args.overlay_flux_style,
            args.overlay_gradient_gamma,
            args.overlay_gradient_min_fraction,
            show_scalar_bar=(args.overlay_show_colorbar and col == len(flux_entries) - 1),
            show_legend=(col == len(flux_entries) - 1),
        )
        set_y_up_camera(plotter, bounds)
        if first_bounds is None:
            first_bounds = bounds

    plotter.link_views()
    if first_bounds is not None:
        set_y_up_camera(plotter, first_bounds)
    reference_suffix = ""
    if flux_display.get("threshold_reference") != "merged":
        reference_suffix = f"_{flux_display['threshold_reference']}_ref"
    potential_suffix = "se_am_potential" if potential_mask == "se_am" else "se_potential"
    if args.overlay_flux_style == "gradient":
        flux_style_suffix = (
            f"_gradient_top{float(flux_display['tail_percent']):g}pct"
            f"_gamma{float(args.overlay_gradient_gamma):g}"
        )
    else:
        flux_style_suffix = "_high_only"
    out_path = (
        spec.output_dir
        / f"{spec.output_prefix}_{potential_suffix}_overlay_se_{flux_quantity}{reference_suffix}{flux_style_suffix}.png"
    )
    plotter.screenshot(str(out_path), transparent_background=True)
    plotter.close()
    print(f"Saved {out_path}")
    return out_path


def render_representative30_void_empty(args: argparse.Namespace, pv, spec: DatasetSpec) -> Path:
    if spec.name != "representative30":
        raise ValueError("--render-void-empty is currently defined for representative30 only.")

    entries = [load_representative30_matrix_entry(args, sample) for sample in SAMPLES]
    for entry in entries:
        print(
            f"representative30 {entry['sample']} AM+SE matrix, void empty: "
            f"matrix_fraction={entry['se_fraction'].shape}, downsample={entry['downsample']}, "
            f"spacing={entry['spacing_um']:.3g} um"
        )

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(shape=(1, len(entries)), off_screen=True, window_size=(args.window_width, args.window_height))
    plotter.set_background("white")
    first_bounds = None
    for col, entry in enumerate(entries):
        plotter.subplot(0, col)
        grid = make_grid(pv, entry["scalar"], entry["se_fraction"], entry["spacing_um"])
        grid.point_data["void_fraction"] = entry["void_fraction"].ravel(order="F")
        bounds = add_void_empty_matrix_mesh(
            plotter,
            pv,
            grid,
            f"{entry['sample']} AM+SE matrix / void empty",
            args.se_threshold,
            args.matrix_color,
            args.matrix_opacity,
            args.void_cut_face_color,
            args.void_cut_face_opacity,
            not args.no_void_cut_faces,
            not args.no_void_empty_cutaway,
        )
        set_y_up_camera(plotter, bounds)
        if first_bounds is None:
            first_bounds = bounds

    plotter.link_views()
    if first_bounds is not None:
        set_y_up_camera(plotter, first_bounds)
    face_suffix = "" if args.no_void_cut_faces else "_cut_faces"
    if args.no_void_empty_cutaway:
        face_suffix = f"{face_suffix}_full_cube"
    out_path = spec.output_dir / f"{spec.output_prefix}_am_se_matrix_void_empty{face_suffix}.png"
    plotter.screenshot(str(out_path), transparent_background=True)
    plotter.close()
    print(f"Saved {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    if args.output_dir and args.dataset == "all":
        raise ValueError("--output-dir is ambiguous with --dataset all; use dataset-specific output dir options.")
    pv = require_pyvista()
    pv.OFF_SCREEN = True

    rendered = []
    for spec in dataset_specs(args):
        if args.render_void_empty:
            if spec.name != "representative30":
                continue
            rendered.append(render_representative30_void_empty(args, pv, spec))
        elif args.overlay_potential_high_flux:
            flux_quantities = [q for q in quantities_for_dataset(spec.name, args.quantity) if q in FLUX_QUANTITIES]
            if not flux_quantities:
                raise ValueError("--overlay-potential-high-flux requires a flux quantity, not potential.")
            overlay_masks = ["se", "se_am"] if args.overlay_potential_mask == "both" else [args.overlay_potential_mask]
            for quantity in flux_quantities:
                for potential_mask in overlay_masks:
                    rendered.append(render_potential_high_flux_overlay(args, pv, spec, quantity, potential_mask))
        else:
            for quantity in quantities_for_dataset(spec.name, args.quantity):
                rendered.append(render_cutaway(args, pv, spec, quantity))
    print("\nRendered 3D cutaway files:")
    for path in rendered:
        print(f"- {path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

from viz_style import apply_publication_style, save_figure, style_axes


SAMPLES = ("WM", "PFDT")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASES = (
    ("CAM", 1, "#376795"),
    ("SE-rich", 2, "#72bcd5"),
    ("Void/carbon-rich", 3, "#ffd06f"),
)
CONTACT_SPECS = (
    ("AM-SE contact", 2, "#72bcd5"),
    ("AM-Void contact", 3, "#ffd06f"),
)
AM_BASE_COLOR = "#6f849b"
SAMPLE_COLORS = {
    "WM": "#376795",
    "PFDT": "#72bcd5",
}
COMBINED_METRIC_GRADIENTS = {
    "WM": ("#82ACD2", "#689BCA"),
    "PFDT": ("#F4AFAC", "#EC716A"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render phase/contact geometry maps and quantify phase-interface metrics."
    )
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument(
        "--representative-root",
        default=PROJECT_ROOT / "server_results" / "30um-in-plane" / "results",
    )
    parser.add_argument(
        "--dataset",
        choices=["representative30", "whole_roi", "all"],
        default="all",
        help="Dataset geometry to analyze.",
    )
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "viz" / "geometry_3d")
    parser.add_argument("--metrics-dir", default=PROJECT_ROOT / "viz" / "geometry_metrics")
    parser.add_argument(
        "--metrics-output-suffix",
        default="",
        help="Optional suffix for metrics outputs, e.g. majority_r2, to avoid overwriting existing figures.",
    )
    parser.add_argument("--voxel-size-um", type=float, default=0.07)
    parser.add_argument(
        "--majority-contact-radius",
        type=int,
        default=2,
        help="Voxel radius for AM surface SE/void majority contact-state metrics.",
    )
    parser.add_argument(
        "--error-style",
        choices=["sd", "faint", "none"],
        default="sd",
        help="Error-bar style for spatial subvolume summary plots.",
    )
    parser.add_argument("--representative-render-downsample", type=int, default=4)
    parser.add_argument("--whole-roi-render-downsample", type=int, default=6)
    parser.add_argument("--representative-connectivity-downsample", type=int, default=2)
    parser.add_argument("--whole-roi-connectivity-downsample", type=int, default=4)
    parser.add_argument("--crop-fraction", type=float, default=1.0)
    parser.add_argument("--window-width", type=int, default=2200)
    parser.add_argument("--window-height", type=int, default=1500)
    parser.add_argument(
        "--contact-render-threshold",
        type=float,
        default=0.03,
        help="Minimum downsampled contact fraction shown in 3D contact maps.",
    )
    parser.add_argument(
        "--render-style",
        choices=["flat", "soft", "strong"],
        default="strong",
        help="3D material preset. soft/strong enable lighting; flat keeps the old unlit rendering.",
    )
    parser.add_argument("--skip-render", action="store_true")
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


def representative_metadata(args: argparse.Namespace, sample: str) -> dict:
    return read_json(Path(args.representative_root) / sample / "representative_flux" / "metadata.json")


def representative_subvolume_rows(args: argparse.Namespace, sample: str) -> list[dict]:
    subvolumes_path = Path(args.representative_root) / sample / "subvolumes.csv"
    if not subvolumes_path.exists():
        meta = representative_metadata(args, sample)
        coords = meta.get("coordinates", {})
        if coords:
            return [{"subvolume_id": meta.get("subvolume_id", "representative"), **coords}]
        return []
    with subvolumes_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def crop_label_to_subvolume_row(label_zyx: np.ndarray, row: dict) -> np.ndarray:
    return label_zyx[
        int(row["z0"]): int(row["z1"]),
        int(row["y0"]): int(row["y1"]),
        int(row["x0"]): int(row["x1"]),
    ]


def crop_label_to_metadata(label_zyx: np.ndarray, meta: dict) -> np.ndarray:
    coords = meta.get("coordinates")
    if not coords:
        return label_zyx
    return label_zyx[
        int(coords["z0"]): int(coords["z1"]),
        int(coords["y0"]): int(coords["y1"]),
        int(coords["x0"]): int(coords["x1"]),
    ]


def center_crop(array: np.ndarray, fraction: float) -> np.ndarray:
    if fraction >= 0.999:
        return array
    slices = []
    for dim in array.shape:
        n = max(8, int(round(dim * fraction)))
        start = (dim - n) // 2
        slices.append(slice(start, start + n))
    return array[tuple(slices)]


def load_label_zyx(args: argparse.Namespace, dataset: str, sample: str) -> np.ndarray:
    label_zyx = tiff.imread(label_path(Path(args.project_root), sample))
    if dataset == "representative30":
        label_zyx = crop_label_to_metadata(label_zyx, representative_metadata(args, sample))
    label_zyx = center_crop(label_zyx, args.crop_fraction)
    return label_zyx.astype(np.uint8, copy=False)


def iter_metric_label_zyx(
    args: argparse.Namespace, dataset: str, sample: str
) -> list[tuple[str, dict[str, int], np.ndarray]]:
    label_zyx = tiff.imread(label_path(Path(args.project_root), sample)).astype(np.uint8, copy=False)
    if dataset != "representative30":
        label_zyx = center_crop(label_zyx, args.crop_fraction)
        return [("whole_roi", {}, label_zyx)]

    metric_labels = []
    for row in representative_subvolume_rows(args, sample):
        region_id = str(row.get("subvolume_id") or "representative")
        coords = {key: int(row[key]) for key in ("x0", "x1", "y0", "y1", "z0", "z1") if key in row}
        subvolume_zyx = crop_label_to_subvolume_row(label_zyx, row)
        subvolume_zyx = center_crop(subvolume_zyx, args.crop_fraction)
        metric_labels.append((region_id, coords, subvolume_zyx))
    if not metric_labels:
        meta = representative_metadata(args, sample)
        label_zyx = crop_label_to_metadata(label_zyx, meta)
        label_zyx = center_crop(label_zyx, args.crop_fraction)
        metric_labels.append((str(meta.get("subvolume_id", "representative")), {}, label_zyx))
    return metric_labels


def zyx_to_xyz(label_zyx: np.ndarray) -> np.ndarray:
    return np.transpose(label_zyx, (2, 1, 0))


def block_fraction(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return mask.astype(np.float32, copy=False)
    crop_shape = tuple((dim // factor) * factor for dim in mask.shape)
    cropped = mask[tuple(slice(0, n) for n in crop_shape)].astype(np.float32, copy=False)
    reshaped = cropped.reshape(
        crop_shape[0] // factor,
        factor,
        crop_shape[1] // factor,
        factor,
        crop_shape[2] // factor,
        factor,
    )
    return reshaped.mean(axis=(1, 3, 5), dtype=np.float32)


def block_any(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return mask.astype(np.float32, copy=False)
    crop_shape = tuple((dim // factor) * factor for dim in mask.shape)
    cropped = mask[tuple(slice(0, n) for n in crop_shape)].astype(np.float32, copy=False)
    reshaped = cropped.reshape(
        crop_shape[0] // factor,
        factor,
        crop_shape[1] // factor,
        factor,
        crop_shape[2] // factor,
        factor,
    )
    return reshaped.max(axis=(1, 3, 5)).astype(np.float32, copy=False)


def interface_face_count(label_zyx: np.ndarray, a: int, b: int) -> int:
    count = 0
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        u = label_zyx[tuple(left)]
        v = label_zyx[tuple(right)]
        count += int(np.count_nonzero(((u == a) & (v == b)) | ((u == b) & (v == a))))
    return count


def boundary_face_count(mask: np.ndarray) -> int:
    count = 0
    for axis in range(3):
        front = [slice(None)] * 3
        back = [slice(None)] * 3
        front[axis] = 0
        back[axis] = -1
        count += int(np.count_nonzero(mask[tuple(front)]))
        count += int(np.count_nonzero(mask[tuple(back)]))
    return count


def local_phase_count(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.uint16, copy=False)
    try:
        from scipy import ndimage

        kernel = np.ones((2 * radius + 1, 2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        return ndimage.convolve(mask.astype(np.uint8, copy=False), kernel, mode="constant", cval=0).astype(
            np.uint16, copy=False
        )
    except ImportError:
        out = np.zeros(mask.shape, dtype=np.uint16)
        padded = np.pad(mask.astype(np.uint8, copy=False), radius, mode="constant")
        for dz in range(2 * radius + 1):
            for dy in range(2 * radius + 1):
                for dx in range(2 * radius + 1):
                    out += padded[
                        dz : dz + mask.shape[0],
                        dy : dy + mask.shape[1],
                        dx : dx + mask.shape[2],
                    ]
        return out


def majority_contact_counts(label_zyx: np.ndarray, labels: dict[str, int], radius: int) -> dict[str, int]:
    am = label_zyx == labels["CAM"]
    surface = am & six_neighbor(~am)
    se_count = local_phase_count(label_zyx == labels["SE-rich"], radius)
    void_count = local_phase_count(label_zyx == labels["Void/carbon-rich"], radius)
    classified = surface & ((se_count + void_count) > 0)
    void_exposed = classified & (void_count > se_count)
    se_covered = classified & ~void_exposed
    other = surface & ~classified
    return {
        "surface": int(np.count_nonzero(surface)),
        "se": int(np.count_nonzero(se_covered)),
        "void": int(np.count_nonzero(void_exposed)),
        "other": int(np.count_nonzero(other)),
    }


def largest_component_fraction(mask: np.ndarray) -> float:
    try:
        from scipy import ndimage
    except ImportError:
        return float("nan")
    if not np.any(mask):
        return 0.0
    structure = ndimage.generate_binary_structure(3, 1)
    components, n_components = ndimage.label(mask, structure=structure)
    if n_components == 0:
        return 0.0
    counts = np.bincount(components.ravel())
    largest = int(counts[1:].max()) if counts.size > 1 else 0
    return float(largest / np.count_nonzero(mask))


def percolates_axis(mask: np.ndarray, axis: int) -> bool:
    try:
        from scipy import ndimage
    except ImportError:
        return False
    if not np.any(mask):
        return False
    structure = ndimage.generate_binary_structure(3, 1)
    components, _ = ndimage.label(mask, structure=structure)
    lo = [slice(None)] * 3
    hi = [slice(None)] * 3
    lo[axis] = 0
    hi[axis] = -1
    lo_labels = set(np.unique(components[tuple(lo)])) - {0}
    hi_labels = set(np.unique(components[tuple(hi)])) - {0}
    return bool(lo_labels & hi_labels)


def compute_metrics(
    args: argparse.Namespace,
    dataset: str,
    sample: str,
    label_zyx: np.ndarray,
    region_id: str,
    region_coords: dict[str, int] | None = None,
) -> dict:
    voxel_size = float(args.voxel_size_um)
    voxel_volume = voxel_size ** 3
    face_area = voxel_size ** 2
    total_voxels = int(label_zyx.size)
    row: dict[str, int | float | str | bool] = {
        "dataset": dataset,
        "sample": sample,
        "region_id": region_id,
        "shape_zyx": "x".join(str(v) for v in label_zyx.shape),
        "voxel_size_um": voxel_size,
        "voxel_count": total_voxels,
        "volume_um3": float(total_voxels * voxel_volume),
    }
    for key, value in (region_coords or {}).items():
        row[key] = value
    labels = {name: value for name, value, _ in PHASES}
    for name, value in labels.items():
        count = int(np.count_nonzero(label_zyx == value))
        key = phase_key(name)
        row[f"{key}_voxel_count"] = count
        row[f"{key}_volume_fraction"] = float(count / total_voxels)
        row[f"{key}_volume_um3"] = float(count * voxel_volume)

    interface_pairs = (
        ("am_se", labels["CAM"], labels["SE-rich"]),
        ("am_void", labels["CAM"], labels["Void/carbon-rich"]),
        ("se_void", labels["SE-rich"], labels["Void/carbon-rich"]),
    )
    for key, a, b in interface_pairs:
        faces = interface_face_count(label_zyx, a, b)
        row[f"{key}_face_count"] = faces
        row[f"{key}_area_um2"] = float(faces * face_area)
        row[f"{key}_area_density_um2_per_um3"] = float(faces * face_area / row["volume_um3"])

    am_internal_surface = float(row["am_se_area_um2"] + row["am_void_area_um2"])
    row["am_internal_surface_area_um2"] = am_internal_surface
    row["am_se_area_fraction_of_am_internal_surface"] = safe_ratio(row["am_se_area_um2"], am_internal_surface)
    row["am_void_area_fraction_of_am_internal_surface"] = safe_ratio(row["am_void_area_um2"], am_internal_surface)

    am_mask = label_zyx == labels["CAM"]
    am_boundary_area = boundary_face_count(am_mask) * face_area
    row["am_boundary_area_um2"] = float(am_boundary_area)
    row["am_surface_plus_roi_boundary_area_um2"] = float(am_internal_surface + am_boundary_area)
    row["am_se_area_fraction_of_am_surface_plus_boundary"] = safe_ratio(
        row["am_se_area_um2"], row["am_surface_plus_roi_boundary_area_um2"]
    )
    row["am_void_area_fraction_of_am_surface_plus_boundary"] = safe_ratio(
        row["am_void_area_um2"], row["am_surface_plus_roi_boundary_area_um2"]
    )

    majority_counts = majority_contact_counts(label_zyx, labels, int(args.majority_contact_radius))
    majority_total = float(majority_counts["se"] + majority_counts["void"])
    majority_surface = float(majority_counts["surface"])
    prefix = f"am_surface_majority_r{int(args.majority_contact_radius)}"
    row[f"{prefix}_contact_radius_voxels"] = int(args.majority_contact_radius)
    row[f"{prefix}_voxel_count"] = majority_counts["surface"]
    row[f"{prefix}_se_voxel_count"] = majority_counts["se"]
    row[f"{prefix}_void_voxel_count"] = majority_counts["void"]
    row[f"{prefix}_other_voxel_count"] = majority_counts["other"]
    row[f"{prefix}_se_fraction_of_classified_contact"] = safe_ratio(majority_counts["se"], majority_total)
    row[f"{prefix}_void_fraction_of_classified_contact"] = safe_ratio(majority_counts["void"], majority_total)
    row[f"{prefix}_se_fraction_of_am_surface"] = safe_ratio(majority_counts["se"], majority_surface)
    row[f"{prefix}_void_fraction_of_am_surface"] = safe_ratio(majority_counts["void"], majority_surface)

    connectivity_factor = (
        args.representative_connectivity_downsample
        if dataset == "representative30"
        else args.whole_roi_connectivity_downsample
    )
    label_conn = label_zyx
    if connectivity_factor > 1:
        label_conn = label_zyx[
            tuple(slice(0, (dim // connectivity_factor) * connectivity_factor) for dim in label_zyx.shape)
        ]
        label_conn = label_conn.reshape(
            label_conn.shape[0] // connectivity_factor,
            connectivity_factor,
            label_conn.shape[1] // connectivity_factor,
            connectivity_factor,
            label_conn.shape[2] // connectivity_factor,
            connectivity_factor,
        )
        # Nearest-center sampling keeps the labels discrete for connectivity checks.
        center = connectivity_factor // 2
        label_conn = label_conn[:, center, :, center, :, center]

    row["connectivity_downsample_factor"] = int(connectivity_factor)
    for name, value in labels.items():
        key = phase_key(name)
        mask = label_conn == value
        row[f"{key}_largest_component_fraction"] = largest_component_fraction(mask)
        row[f"{key}_percolates_x"] = percolates_axis(mask, 2)
        row[f"{key}_percolates_y"] = percolates_axis(mask, 1)
        row[f"{key}_percolates_z"] = percolates_axis(mask, 0)
    return row


def phase_key(name: str) -> str:
    return name.lower().replace("/", "_").replace("-", "_").replace(" ", "_").replace("rich", "").strip("_")


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def make_grid(pv, arrays: dict[str, np.ndarray], spacing_um: float):
    shape = next(iter(arrays.values())).shape
    grid = pv.ImageData()
    grid.dimensions = shape
    grid.spacing = (spacing_um, spacing_um, spacing_um)
    for name, array in arrays.items():
        grid.point_data[name] = array.ravel(order="F")
    return grid


def material_args(args: argparse.Namespace, *, opacity: float) -> dict:
    if args.render_style == "flat":
        return {
            "opacity": opacity,
            "smooth_shading": False,
            "lighting": False,
        }
    if args.render_style == "strong":
        return {
            "opacity": opacity,
            "smooth_shading": True,
            "lighting": True,
            "ambient": 0.20,
            "diffuse": 0.88,
            "specular": 0.30,
            "specular_power": 22,
        }
    return {
        "opacity": opacity,
        "smooth_shading": True,
        "lighting": True,
        "ambient": 0.34,
        "diffuse": 0.74,
        "specular": 0.16,
        "specular_power": 16,
    }


def configure_lighting(plotter, pv, args: argparse.Namespace) -> None:
    if args.render_style == "flat":
        return
    try:
        plotter.enable_lightkit()
    except Exception:
        return
    if args.render_style == "strong":
        try:
            plotter.add_light(pv.Light(position=(1, 1.5, 1.7), focal_point=(0, 0, 0), intensity=0.55))
        except Exception:
            pass


def cutaway_bounds_from_grid(pv, grid, scalar_name: str, threshold: float = 0.05):
    base = grid.threshold(value=threshold, scalars=scalar_name)
    bounds = base.bounds
    lx = bounds[1] - bounds[0]
    ly = bounds[3] - bounds[2]
    lz = bounds[5] - bounds[4]
    clip_bounds = (
        bounds[0] + 0.42 * lx,
        bounds[1],
        bounds[2] + 0.42 * ly,
        bounds[3],
        bounds[4] + 0.42 * lz,
        bounds[5],
    )
    return base.clip_box(clip_bounds, invert=True), bounds


def add_outline_axes_title(plotter, pv, bounds, title: str) -> None:
    outline = pv.Box(bounds=bounds).outline()
    plotter.add_mesh(outline, color="black", line_width=0.9)
    plotter.add_axes(line_width=1, labels_off=False)
    plotter.add_text(title, position=(0.03, 0.92), font_size=10, viewport=True)


def add_phase_panel(plotter, pv, args: argparse.Namespace, grid, title: str, phase_names: tuple[str, ...]) -> tuple:
    cut, bounds = cutaway_bounds_from_grid(pv, grid, "solid_fraction")
    overview_opacity = {"CAM": 0.50, "SE-rich": 0.46, "Void/carbon-rich": 0.38}
    single_opacity = {"CAM": 0.88, "SE-rich": 0.84, "Void/carbon-rich": 0.78}
    for name, _, color in PHASES:
        if name not in phase_names:
            continue
        phase = cut.threshold(value=0.12, scalars=phase_key(name), method="upper", all_scalars=True)
        if phase.n_points:
            opacity = overview_opacity[name] if len(phase_names) > 1 else single_opacity[name]
            plotter.add_mesh(
                phase,
                color=color,
                show_scalar_bar=False,
                **material_args(args, opacity=opacity),
            )
    add_outline_axes_title(plotter, pv, bounds, title)
    return bounds


def add_contact_panel(plotter, pv, args: argparse.Namespace, grid, title: str, contact_name: str, color: str) -> tuple:
    cut, bounds = cutaway_bounds_from_grid(pv, grid, "am_fraction")
    plotter.add_mesh(
        cut,
        color=AM_BASE_COLOR,
        show_scalar_bar=False,
        **material_args(args, opacity=0.30),
    )
    contact = cut.threshold(value=args.contact_render_threshold, scalars=contact_name, method="upper", all_scalars=True)
    if contact.n_points:
        plotter.add_mesh(
            contact,
            color=color,
            show_scalar_bar=False,
            **material_args(args, opacity=0.96),
        )
    add_outline_axes_title(plotter, pv, bounds, title)
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
    distance = float(np.linalg.norm(span) * 2.20)
    direction = np.array([1.0, 0.78, 1.0], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    plotter.camera.position = tuple(center + distance * direction)
    plotter.camera.focal_point = tuple(center)
    plotter.camera.view_up = (0.0, 1.0, 0.0)
    plotter.camera.zoom(0.98)


def render_phase_overview(args: argparse.Namespace, pv, dataset: str, render_data: dict[str, dict]) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(shape=(len(SAMPLES), 4), off_screen=True, window_size=(args.window_width, args.window_height))
    plotter.set_background("white")
    configure_lighting(plotter, pv, args)
    first_bounds = None
    columns: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("Overview", tuple(name for name, _, _ in PHASES)),
        ("CAM", ("CAM",)),
        ("SE-rich", ("SE-rich",)),
        ("Void/carbon-rich", ("Void/carbon-rich",)),
    )
    for row, sample in enumerate(SAMPLES):
        for col, (title, phase_names) in enumerate(columns):
            plotter.subplot(row, col)
            bounds = add_phase_panel(
                plotter, pv, args, render_data[sample]["phase_grid"], f"{sample} {title}", phase_names
            )
            set_y_up_camera(plotter, bounds)
            if first_bounds is None:
                first_bounds = bounds
    plotter.link_views()
    if first_bounds is not None:
        set_y_up_camera(plotter, first_bounds)
    out_path = out_dir / f"{dataset}_phase_overview_3d.png"
    plotter.screenshot(str(out_path), transparent_background=True)
    plotter.close()
    print(f"Saved {out_path}")
    return out_path


def render_contact_maps(args: argparse.Namespace, pv, dataset: str, render_data: dict[str, dict]) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(
        shape=(len(SAMPLES), len(CONTACT_SPECS)), off_screen=True, window_size=(args.window_width, args.window_height)
    )
    plotter.set_background("white")
    configure_lighting(plotter, pv, args)
    first_bounds = None
    for row, sample in enumerate(SAMPLES):
        for col, (contact_title, _, color) in enumerate(CONTACT_SPECS):
            plotter.subplot(row, col)
            bounds = add_contact_panel(
                plotter,
                pv,
                args,
                render_data[sample]["contact_grid"],
                f"{sample} {contact_title}",
                contact_key(contact_title),
                color,
            )
            set_y_up_camera(plotter, bounds)
            if first_bounds is None:
                first_bounds = bounds
    plotter.link_views()
    if first_bounds is not None:
        set_y_up_camera(plotter, first_bounds)
    out_path = out_dir / f"{dataset}_am_contact_adjacency_3d.png"
    plotter.screenshot(str(out_path), transparent_background=True)
    plotter.close()
    print(f"Saved {out_path}")
    return out_path


def contact_key(title: str) -> str:
    return title.lower().replace("-", "_").replace(" ", "_")


def prepare_render_data(args: argparse.Namespace, pv, dataset: str, labels: dict[str, np.ndarray]) -> dict[str, dict]:
    factor = args.representative_render_downsample if dataset == "representative30" else args.whole_roi_render_downsample
    spacing_um = float(args.voxel_size_um) * int(factor)
    out = {}
    for sample, label_zyx in labels.items():
        label_xyz = zyx_to_xyz(label_zyx)
        phase_arrays = {}
        for name, value, _ in PHASES:
            phase_arrays[phase_key(name)] = block_fraction(label_xyz == value, factor)
        phase_arrays["solid_fraction"] = np.clip(
            phase_arrays["cam"] + phase_arrays["se"] + phase_arrays["void_carbon"], 0.0, 1.0
        )
        phase_grid = make_grid(pv, phase_arrays, spacing_um)

        am = label_xyz == 1
        contact_arrays = {"am_fraction": block_fraction(am, factor)}
        for title, neighbor_label, _ in CONTACT_SPECS:
            contact = am & six_neighbor(label_xyz == neighbor_label)
            contact_arrays[contact_key(title)] = block_fraction(contact, factor)
        contact_grid = make_grid(pv, contact_arrays, spacing_um)
        out[sample] = {"phase_grid": phase_grid, "contact_grid": contact_grid}
    return out


def six_neighbor(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=bool)
    out[1:, :, :] |= mask[:-1, :, :]
    out[:-1, :, :] |= mask[1:, :, :]
    out[:, 1:, :] |= mask[:, :-1, :]
    out[:, :-1, :] |= mask[:, 1:, :]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


def write_metrics(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    leading = [
        "dataset",
        "sample",
        "region_id",
        "x0",
        "x1",
        "y0",
        "y1",
        "z0",
        "z1",
        "shape_zyx",
        "voxel_size_um",
        "voxel_count",
        "volume_um3",
    ]
    fieldnames = leading + [key for key in fieldnames if key not in leading]

    def write_csv(path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    try:
        write_csv(out_path)
        saved_path = out_path
    except PermissionError:
        saved_path = out_path.with_name(f"{out_path.stem}_latest{out_path.suffix}")
        write_csv(saved_path)
        print(f"Could not overwrite locked file {out_path}; wrote fallback copy instead.")
    print(f"Saved {saved_path}")


def error_kwargs_for_style(error_style: str) -> dict:
    if error_style == "none":
        return {"show": False, "error_kw": {}}
    alpha = 0.18 if error_style == "faint" else 1.0
    return {
        "show": True,
        "error_kw": {"elinewidth": 0.9, "capthick": 0.9, "alpha": alpha},
    }


def plot_contact_metric_summary(rows: list[dict], out_path: Path, error_style: str = "sd") -> None:
    apply_publication_style()
    metrics = [
        ("am_se_area_fraction_of_am_internal_surface", "AM-SE / AM surface"),
        ("am_void_area_fraction_of_am_internal_surface", "AM-Void / AM surface"),
        ("se_largest_component_fraction", "SE largest component"),
    ]
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    dataset_labels = {"representative30": "30 um", "whole_roi": "whole ROI"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.1, 2.4), constrained_layout=True)
    x = np.arange(len(datasets))
    width = 0.34
    error_style_kwargs = error_kwargs_for_style(error_style)
    for ax, (key, label) in zip(np.ravel(axes), metrics):
        for offset, sample in [(-width / 2, "WM"), (width / 2, "PFDT")]:
            means = []
            errors = []
            grouped_values = []
            for dataset in datasets:
                values = [
                    float(row[key])
                    for row in rows
                    if row["dataset"] == dataset
                    and row["sample"] == sample
                    and key in row
                    and np.isfinite(float(row[key]))
                ]
                grouped_values.append(values)
                means.append(float(np.mean(values)) if values else float("nan"))
                errors.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=errors if error_style_kwargs["show"] else None,
                capsize=2.5 if error_style_kwargs["show"] else 0,
                color=SAMPLE_COLORS[sample],
                label=sample,
                error_kw=error_style_kwargs["error_kw"],
            )
            for i, values in enumerate(grouped_values):
                if len(values) <= 1:
                    continue
                jitter = np.linspace(-0.045, 0.045, len(values))
                ax.scatter(
                    np.full(len(values), x[i] + offset) + jitter,
                    values,
                    s=9,
                    color="#2f2f2f",
                    alpha=0.55,
                    linewidths=0,
                    zorder=3,
                )
        ax.set_xticks(x, [dataset_labels.get(dataset, dataset) for dataset in datasets])
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
        style_axes(ax, minor=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.55), ncol=1, frameon=False)
    save_figure(fig, out_path)
    print(f"Saved {out_path}")


def plot_majority_contact_metric_summary(
    rows: list[dict], out_path: Path, radius: int, error_style: str = "sd"
) -> None:
    apply_publication_style()
    prefix = f"am_surface_majority_r{int(radius)}"
    metrics = [
        (f"{prefix}_se_fraction_of_am_surface", "AM-SE / AM surface"),
        (f"{prefix}_void_fraction_of_am_surface", "AM-Void / AM surface"),
        ("se_largest_component_fraction", "SE largest component"),
    ]
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    dataset_labels = {"representative30": "30 um", "whole_roi": "whole ROI"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.1, 2.4), constrained_layout=True)
    x = np.arange(len(datasets))
    width = 0.34
    error_style_kwargs = error_kwargs_for_style(error_style)
    for ax, (key, label) in zip(np.ravel(axes), metrics):
        for offset, sample in [(-width / 2, "WM"), (width / 2, "PFDT")]:
            means = []
            errors = []
            grouped_values = []
            for dataset in datasets:
                values = [
                    float(row[key])
                    for row in rows
                    if row["dataset"] == dataset
                    and row["sample"] == sample
                    and key in row
                    and np.isfinite(float(row[key]))
                ]
                grouped_values.append(values)
                means.append(float(np.mean(values)) if values else float("nan"))
                errors.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=errors if error_style_kwargs["show"] else None,
                capsize=2.5 if error_style_kwargs["show"] else 0,
                color=SAMPLE_COLORS[sample],
                label=sample,
                error_kw=error_style_kwargs["error_kw"],
            )
            for i, values in enumerate(grouped_values):
                if len(values) <= 1:
                    continue
                jitter = np.linspace(-0.045, 0.045, len(values))
                ax.scatter(
                    np.full(len(values), x[i] + offset) + jitter,
                    values,
                    s=9,
                    color="#2f2f2f",
                    alpha=0.55,
                    linewidths=0,
                    zorder=3,
                )
        ax.set_xticks(x, [dataset_labels.get(dataset, dataset) for dataset in datasets])
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
        style_axes(ax, minor=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.55), ncol=1, frameon=False)
    save_figure(fig, out_path)
    print(f"Saved {out_path}")


def hex_to_rgb01(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.array([int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=float)


def add_gradient_bar(ax, x_center: float, height: float, width: float, bottom_color: str, top_color: str) -> None:
    left = x_center - width / 2
    rect = plt.Rectangle((left, 0), width, height, facecolor="none", edgecolor="none")
    ax.add_patch(rect)
    bottom = hex_to_rgb01(bottom_color)
    top = hex_to_rgb01(top_color)
    gradient = np.linspace(bottom, top, 256).reshape(256, 1, 3)
    im = ax.imshow(
        gradient,
        extent=[left, left + width, 0, height],
        origin="lower",
        aspect="auto",
        interpolation="bicubic",
        zorder=2,
    )
    im.set_clip_path(rect)


def plot_representative30_combined_contact_summary(rows: list[dict], out_path: Path, radius: int) -> None:
    apply_publication_style()
    prefix = f"am_surface_majority_r{int(radius)}"
    metrics = [
        ("void_carbon_volume_fraction", "Apparent\nvoid/C-rich"),
        (f"{prefix}_se_fraction_of_am_surface", "SE-covered\nAM"),
        (f"{prefix}_void_fraction_of_am_surface", "Void/C-exposed\nAM"),
    ]
    rows = [row for row in rows if row["dataset"] == "representative30"]
    samples = ["WM", "PFDT"]
    x = np.arange(len(metrics), dtype=float)
    width = 0.30
    offsets = {"WM": -width / 1.85, "PFDT": width / 1.85}
    fig, ax = plt.subplots(figsize=(66.80 / 25.4, 53.97 / 25.4), constrained_layout=False)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.27, top=0.96)
    for sample in samples:
        bottom_color, top_color = COMBINED_METRIC_GRADIENTS[sample]
        for i, (key, _) in enumerate(metrics):
            values = np.array(
                [
                    float(row[key])
                    for row in rows
                    if row["sample"] == sample
                    and key in row
                    and np.isfinite(float(row[key]))
                ],
                dtype=float,
            )
            if values.size == 0:
                continue
            xpos = x[i] + offsets[sample]
            add_gradient_bar(ax, xpos, float(np.mean(values)), width, bottom_color, top_color)
            jitter = np.linspace(-0.035, 0.035, values.size)
            ax.scatter(
                np.full(values.size, xpos) + jitter,
                values,
                s=8,
                color="#303030",
                alpha=0.72,
                linewidths=0,
                zorder=4,
            )
    ax.set_xlim(-0.52, len(metrics) - 0.48)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("Fraction")
    style_axes(ax, minor=False)
    ax.tick_params(axis="x", labelsize=8, pad=2)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COMBINED_METRIC_GRADIENTS["WM"][1], edgecolor="none", label="WM"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COMBINED_METRIC_GRADIENTS["PFDT"][1], edgecolor="none", label="PFDT"),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=2, handlelength=1.1, columnspacing=0.9)
    fig.savefig(
        out_path,
        dpi=600,
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    plt.close(fig)
    print(f"Saved {out_path}")


def datasets_to_run(requested: str) -> list[str]:
    if requested == "all":
        return ["representative30", "whole_roi"]
    return [requested]


def main() -> None:
    args = parse_args()
    pv = None if args.skip_render else require_pyvista()
    if pv is not None:
        pv.OFF_SCREEN = True

    all_rows = []
    for dataset in datasets_to_run(args.dataset):
        for sample in SAMPLES:
            for region_id, region_coords, label_zyx in iter_metric_label_zyx(args, dataset, sample):
                print(f"{dataset} {sample} {region_id}: label shape zyx={label_zyx.shape}")
                all_rows.append(compute_metrics(args, dataset, sample, label_zyx, region_id, region_coords))
        if pv is not None:
            labels = {sample: load_label_zyx(args, dataset, sample) for sample in SAMPLES}
            render_data = prepare_render_data(args, pv, dataset, labels)
            render_phase_overview(args, pv, dataset, render_data)
            render_contact_maps(args, pv, dataset, render_data)

    metrics_dir = Path(args.metrics_dir)
    suffix = str(args.metrics_output_suffix).strip()
    suffix_token = f"_{suffix}" if suffix else ""
    metrics_path = metrics_dir / f"geometry_metrics_summary{suffix_token}.csv"
    write_metrics(all_rows, metrics_path)
    legacy_plot_name = (
        f"geometry_contact_metrics_summary_legacy{suffix_token}.png"
        if suffix
        else "geometry_contact_metrics_summary.png"
    )
    plot_contact_metric_summary(all_rows, metrics_dir / legacy_plot_name, args.error_style)
    majority_plot_stem = f"geometry_contact_metrics_summary_majority_r{int(args.majority_contact_radius)}"
    majority_plot_name = f"{majority_plot_stem}{suffix_token}.png"
    plot_majority_contact_metric_summary(
        all_rows,
        metrics_dir / majority_plot_name,
        int(args.majority_contact_radius),
        args.error_style,
    )
    combined_plot_name = (
        f"geometry_contact_metrics_summary_majority_r{int(args.majority_contact_radius)}_30um_combined{suffix_token}.png"
    )
    plot_representative30_combined_contact_summary(
        all_rows,
        metrics_dir / combined_plot_name,
        int(args.majority_contact_radius),
    )


if __name__ == "__main__":
    main()

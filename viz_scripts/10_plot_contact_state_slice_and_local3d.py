from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from matplotlib.lines import Line2D

from viz_style import apply_publication_style, save_figure, style_axes


SAMPLES = ("WM", "PFDT")
SAMPLE_TITLES = {"WM": "WM-LPSCB", "PFDT": "PFDT@LPSCB"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAM_LABEL = 1
SE_LABEL = 2
VOID_LABEL = 3

PHASE_COLORS = {
    "am": np.array([0.78, 0.80, 0.82]),
    "se": np.array([0.73, 0.90, 0.93]),
    "void": np.array([0.96, 0.82, 0.47]),
    "background": np.array([1.0, 1.0, 1.0]),
}
CONTACT_COLORS = {
    "se": "#15a8c7",
    "void": "#ef9f2d",
}
CONTACT_RGB = {
    "se": np.array([21, 168, 199], dtype=np.float32) / 255.0,
    "void": np.array([239, 159, 45], dtype=np.float32) / 255.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw representative 2D AM contact-state slices and local 3D AM surface exposure views."
    )
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument(
        "--representative-root",
        default=PROJECT_ROOT / "server_results" / "30um-in-plane" / "results",
    )
    parser.add_argument("--output-dir-2d", default=PROJECT_ROOT / "viz" / "am_contact_state_2d")
    parser.add_argument("--output-dir-3d", default=PROJECT_ROOT / "viz" / "am_surface_exposure_local3d")
    parser.add_argument("--voxel-size-um", type=float, default=0.07)
    parser.add_argument("--contact-radius", type=int, default=2)
    parser.add_argument("--slice-axis", choices=["z"], default="z")
    parser.add_argument("--local-sizes-um", nargs="+", type=float, default=[10.0, 15.0])
    parser.add_argument("--local-render-downsample", type=int, default=2)
    parser.add_argument("--window-width", type=int, default=1900)
    parser.add_argument("--window-height", type=int, default=900)
    parser.add_argument("--skip-2d", action="store_true")
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument("--clean-only", action="store_true", help="Only save the title/axis-free 2D overlay.")
    return parser.parse_args()


def require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise SystemExit(
            "PyVista is not installed. Install pyvista/vtk or rerun with --skip-3d."
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


def crop_label_to_metadata(label_zyx: np.ndarray, meta: dict) -> np.ndarray:
    coords = meta.get("coordinates")
    if not coords:
        return label_zyx
    return label_zyx[
        int(coords["z0"]): int(coords["z1"]),
        int(coords["y0"]): int(coords["y1"]),
        int(coords["x0"]): int(coords["x1"]),
    ]


def load_representative_label(args: argparse.Namespace, sample: str) -> tuple[np.ndarray, dict]:
    label_zyx = tiff.imread(label_path(Path(args.project_root), sample)).astype(np.uint8, copy=False)
    meta = representative_metadata(args, sample)
    return crop_label_to_metadata(label_zyx, meta), meta


def six_neighbor(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=bool)
    out[1:, :, :] |= mask[:-1, :, :]
    out[:-1, :, :] |= mask[1:, :, :]
    out[:, 1:, :] |= mask[:, :-1, :]
    out[:, :-1, :] |= mask[:, 1:, :]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


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


def majority_contact_masks(label_zyx: np.ndarray, radius: int) -> dict[str, np.ndarray]:
    surface = (label_zyx == CAM_LABEL) & six_neighbor(label_zyx != CAM_LABEL)
    se_count = local_phase_count(label_zyx == SE_LABEL, radius)
    void_count = local_phase_count(label_zyx == VOID_LABEL, radius)
    classified = surface & ((se_count + void_count) > 0)
    void_exposed = classified & (void_count > se_count)
    se_covered = classified & ~void_exposed
    other = surface & ~classified
    return {"surface": surface, "se": se_covered, "void": void_exposed, "other": other}


def contact_fraction(masks: dict[str, np.ndarray]) -> float:
    total = int(np.count_nonzero(masks["se"]) + np.count_nonzero(masks["void"]))
    if total == 0:
        return float("nan")
    return float(np.count_nonzero(masks["void"]) / total)


def phase_rgb(slice_yx: np.ndarray) -> np.ndarray:
    rgb = np.ones((*slice_yx.shape, 3), dtype=np.float32)
    rgb[:] = PHASE_COLORS["background"]
    rgb[slice_yx == CAM_LABEL] = PHASE_COLORS["am"]
    rgb[slice_yx == SE_LABEL] = PHASE_COLORS["se"]
    rgb[slice_yx == VOID_LABEL] = PHASE_COLORS["void"]
    return rgb


def dilate_2d(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    try:
        from scipy import ndimage

        return ndimage.binary_dilation(mask, iterations=iterations)
    except ImportError:
        out = mask.copy()
        for _ in range(iterations):
            padded = np.pad(out, 1, mode="constant")
            out = (
                padded[1:-1, 1:-1]
                | padded[:-2, 1:-1]
                | padded[2:, 1:-1]
                | padded[1:-1, :-2]
                | padded[1:-1, 2:]
            )
        return out


def contact_overlay(mask: np.ndarray, rgb: np.ndarray, alpha: float = 0.96) -> np.ndarray:
    thick = dilate_2d(mask, iterations=1)
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[thick, :3] = rgb
    rgba[thick, 3] = alpha
    return rgba


def choose_slice(label_zyx: np.ndarray, masks: dict[str, np.ndarray], target_void_fraction: float) -> dict:
    depth = label_zyx.shape[0]
    margin = max(3, depth // 12)
    rows = []
    for z in range(margin, depth - margin):
        se_n = int(np.count_nonzero(masks["se"][z]))
        void_n = int(np.count_nonzero(masks["void"][z]))
        surface_n = se_n + void_n
        if surface_n == 0:
            continue
        void_fraction = void_n / surface_n
        rows.append(
            {
                "z_index": z,
                "surface_pixels": surface_n,
                "se_pixels": se_n,
                "void_pixels": void_n,
                "void_fraction": void_fraction,
            }
        )
    if not rows:
        raise RuntimeError("No AM surface pixels found for slice selection.")
    surface_cutoff = np.percentile([row["surface_pixels"] for row in rows], 60)
    center = (depth - 1) / 2
    best = min(
        rows,
        key=lambda row: (
            abs(row["void_fraction"] - target_void_fraction)
            + (0.15 if row["surface_pixels"] < surface_cutoff else 0.0)
            + 0.02 * abs(row["z_index"] - center) / depth
        ),
    )
    return best


def plot_slice_overlay(samples: dict[str, dict], args: argparse.Namespace) -> Path:
    apply_publication_style()
    out_dir = Path(args.output_dir_2d)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
    for ax, sample in zip(axes, SAMPLES):
        data = samples[sample]
        label_zyx = data["label"]
        masks = data["masks"]
        z = int(data["slice"]["z_index"])
        image = phase_rgb(label_zyx[z])
        ny, nx = label_zyx.shape[1:]
        extent = [0, nx * args.voxel_size_um, 0, ny * args.voxel_size_um]
        ax.imshow(image, origin="lower", extent=extent, interpolation="nearest")
        ax.imshow(
            contact_overlay(masks["se"][z], CONTACT_RGB["se"]),
            origin="lower",
            extent=extent,
            interpolation="nearest",
        )
        ax.imshow(
            contact_overlay(masks["void"][z], CONTACT_RGB["void"]),
            origin="lower",
            extent=extent,
            interpolation="nearest",
        )
        ax.set_title(
            f"{SAMPLE_TITLES[sample]}\nz = {z * args.voxel_size_um:.1f} μm, void {data['slice']['void_fraction']:.1%}"
        )
        ax.set_xlabel("x (μm)")
        ax.set_ylabel("y (μm)")
        style_axes(ax, minor=True)
        add_scale_bar(ax, length_um=5.0)
    legend_items = [
        Line2D([0], [0], color=CONTACT_COLORS["se"], lw=2.0, label="AM boundary adjacent to SE-rich"),
        Line2D([0], [0], color=CONTACT_COLORS["void"], lw=2.0, label="AM boundary adjacent to void/carbon-rich"),
    ]
    fig.legend(handles=legend_items, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    out_path = out_dir / f"representative30_contact_state_slice_overlay_majority_r{args.contact_radius}.png"
    save_figure(fig, out_path)
    print(f"Saved {out_path}")
    plot_slice_overlay_clean(samples, args)
    return out_path


def plot_slice_overlay_clean(samples: dict[str, dict], args: argparse.Namespace) -> Path:
    apply_publication_style()
    out_dir = Path(args.output_dir_2d)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.8), constrained_layout=True)
    for ax, sample in zip(axes, SAMPLES):
        data = samples[sample]
        label_zyx = data["label"]
        masks = data["masks"]
        z = int(data["slice"]["z_index"])
        image = phase_rgb(label_zyx[z])
        ny, nx = label_zyx.shape[1:]
        extent = [0, nx * args.voxel_size_um, 0, ny * args.voxel_size_um]
        ax.imshow(image, origin="lower", extent=extent, interpolation="nearest")
        ax.imshow(
            contact_overlay(masks["se"][z], CONTACT_RGB["se"]),
            origin="lower",
            extent=extent,
            interpolation="nearest",
        )
        ax.imshow(
            contact_overlay(masks["void"][z], CONTACT_RGB["void"]),
            origin="lower",
            extent=extent,
            interpolation="nearest",
        )
        add_scale_bar(ax, length_um=5.0)
        ax.set_axis_off()
    out_path = out_dir / f"representative30_contact_state_slice_overlay_majority_r{args.contact_radius}_clean_scalebar.png"
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def add_scale_bar(ax, length_um: float) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    pad_x = 0.06 * (x1 - x0)
    pad_y = 0.07 * (y1 - y0)
    xb = x1 - pad_x - length_um
    yb = y0 + pad_y
    ax.plot([xb, xb + length_um], [yb, yb], color="black", lw=1.2, solid_capstyle="butt")
    ax.text(xb + length_um / 2, yb + 0.45, f"{length_um:g} μm", ha="center", va="bottom", fontsize=9)


def write_slice_metrics(samples: dict[str, dict], args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir_2d)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"representative30_contact_state_slice_overlay_majority_r{args.contact_radius}_metrics.csv"
    rows = []
    for sample, data in samples.items():
        row = {
            "sample": sample,
            "subvolume_id": data["meta"].get("subvolume_id", ""),
            "z_index_local": data["slice"]["z_index"],
            "z_um_local": data["slice"]["z_index"] * args.voxel_size_um,
            "surface_pixels": data["slice"]["surface_pixels"],
            "se_pixels": data["slice"]["se_pixels"],
            "void_pixels": data["slice"]["void_pixels"],
            "slice_void_fraction": data["slice"]["void_fraction"],
            "volume_void_fraction": data["volume_void_fraction"],
            "contact_radius_voxels": args.contact_radius,
        }
        rows.append(row)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_path}")
    return out_path


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


def count_large_am_components(mask: np.ndarray) -> int:
    try:
        from scipy import ndimage
    except ImportError:
        return 0
    components, n_components = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    if n_components == 0:
        return 0
    counts = np.bincount(components.ravel())[1:]
    min_size = max(300, int(mask.size * 0.004))
    return int(np.count_nonzero(counts >= min_size))


def candidate_starts(dim: int, size: int) -> list[int]:
    if size >= dim:
        return [0]
    step = max(1, size // 4)
    values = list(range(0, dim - size + 1, step))
    values.append(dim - size)
    return sorted(set(values))


def select_local_window(sample: str, data: dict, size_um: float, args: argparse.Namespace) -> dict:
    label = data["label"]
    masks = data["masks"]
    size_vox = int(round(size_um / args.voxel_size_um))
    size_vox = max(16, min(size_vox, min(label.shape)))
    starts_z = candidate_starts(label.shape[0], size_vox)
    starts_y = candidate_starts(label.shape[1], size_vox)
    starts_x = candidate_starts(label.shape[2], size_vox)
    quick = []
    target = data["volume_void_fraction"]
    for z0 in starts_z:
        for y0 in starts_y:
            for x0 in starts_x:
                s = np.s_[z0 : z0 + size_vox, y0 : y0 + size_vox, x0 : x0 + size_vox]
                se_n = int(np.count_nonzero(masks["se"][s]))
                void_n = int(np.count_nonzero(masks["void"][s]))
                surface_n = se_n + void_n
                if surface_n < 200:
                    continue
                void_fraction = void_n / surface_n
                cam_fraction = float(np.count_nonzero(label[s] == CAM_LABEL) / (size_vox**3))
                void_phase_fraction = float(np.count_nonzero(label[s] == VOID_LABEL) / (size_vox**3))
                score = (
                    abs(void_fraction - target)
                    + 0.12 * abs(cam_fraction - 0.45)
                    + 0.08 * max(0.0, 0.01 - void_phase_fraction)
                )
                quick.append(
                    {
                        "sample": sample,
                        "size_um": size_um,
                        "size_voxels": size_vox,
                        "z0": z0,
                        "z1": z0 + size_vox,
                        "y0": y0,
                        "y1": y0 + size_vox,
                        "x0": x0,
                        "x1": x0 + size_vox,
                        "surface_voxels": surface_n,
                        "se_surface_voxels": se_n,
                        "void_surface_voxels": void_n,
                        "void_fraction": void_fraction,
                        "cam_fraction": cam_fraction,
                        "void_phase_fraction": void_phase_fraction,
                        "score": score,
                    }
                )
    if not quick:
        raise RuntimeError(f"No candidate local windows found for {sample} {size_um} um.")
    quick.sort(key=lambda row: row["score"])
    best = None
    for row in quick[:20]:
        s = np.s_[row["z0"] : row["z1"], row["y0"] : row["y1"], row["x0"] : row["x1"]]
        n_components = count_large_am_components(label[s] == CAM_LABEL)
        component_penalty = 0.03 * abs(n_components - 3)
        final_score = row["score"] + component_penalty
        row = {**row, "large_am_component_count": n_components, "final_score": final_score}
        if best is None or row["final_score"] < best["final_score"]:
            best = row
    assert best is not None
    return best


def make_grid(pv, arrays_zyx: dict[str, np.ndarray], spacing_um: float):
    arrays_xyz = {key: np.transpose(value, (2, 1, 0)) for key, value in arrays_zyx.items()}
    first = next(iter(arrays_xyz.values()))
    grid = pv.ImageData()
    grid.dimensions = first.shape
    grid.spacing = (spacing_um, spacing_um, spacing_um)
    for name, array in arrays_xyz.items():
        grid.point_data[name] = array.ravel(order="F")
    return grid


def contour_mesh(grid, scalar: str, threshold: float):
    mesh = grid.contour([threshold], scalars=scalar)
    if mesh.n_points == 0:
        return mesh
    return mesh.clean().compute_normals(auto_orient_normals=True, consistent_normals=True)


def add_mesh_if_present(plotter, mesh, **kwargs) -> None:
    if mesh is not None and mesh.n_points:
        plotter.add_mesh(mesh, show_scalar_bar=False, smooth_shading=True, lighting=True, **kwargs)


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


def local_arrays(data: dict, window: dict, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], float]:
    label = data["label"]
    masks = data["masks"]
    s = np.s_[window["z0"] : window["z1"], window["y0"] : window["y1"], window["x0"] : window["x1"]]
    local_label = label[s]
    local_am = local_label == CAM_LABEL
    local_se_contact = masks["se"][s]
    local_void_contact = masks["void"][s]
    se_phase_context = (local_label == SE_LABEL) & (local_phase_count(local_se_contact, 1) > 0)
    void_phase_context = (local_label == VOID_LABEL) & (local_phase_count(local_void_contact, 1) > 0)
    arrays = {
        "am": block_fraction(local_am, args.local_render_downsample),
        "se_phase_context": block_fraction(se_phase_context, args.local_render_downsample),
        "void_phase_context": block_fraction(void_phase_context, args.local_render_downsample),
        "se_contact": block_fraction(local_se_contact, args.local_render_downsample),
        "void_contact": block_fraction(local_void_contact, args.local_render_downsample),
    }
    return arrays, args.voxel_size_um * args.local_render_downsample


def render_local_3d(samples: dict[str, dict], windows: list[dict], size_um: float, args: argparse.Namespace) -> Path:
    pv = require_pyvista()
    pv.OFF_SCREEN = True
    out_dir = Path(args.output_dir_3d)
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(args.window_width, args.window_height))
    first_bounds = None
    for col, sample in enumerate(SAMPLES):
        plotter.subplot(0, col)
        window = next(row for row in windows if row["sample"] == sample and abs(row["size_um"] - size_um) < 1e-9)
        arrays, spacing = local_arrays(samples[sample], window, args)
        grid = make_grid(pv, arrays, spacing)
        bounds = grid.bounds
        if first_bounds is None:
            first_bounds = bounds
        am_mesh = contour_mesh(grid, "am", 0.45)
        se_phase_mesh = contour_mesh(grid, "se_phase_context", 0.08)
        void_phase_mesh = contour_mesh(grid, "void_phase_context", 0.08)
        contact_mesh = contour_mesh(grid, "void_contact", 0.05)
        add_mesh_if_present(
            plotter,
            se_phase_mesh,
            color="#72bcd5",
            opacity=0.025,
            ambient=0.55,
            diffuse=0.55,
            specular=0.03,
        )
        add_mesh_if_present(
            plotter,
            void_phase_mesh,
            color="#ffd06f",
            opacity=0.16,
            ambient=0.55,
            diffuse=0.55,
            specular=0.03,
        )
        add_mesh_if_present(plotter, am_mesh, color="#c9c9c9", opacity=0.82, ambient=0.34, diffuse=0.72, specular=0.06)
        add_mesh_if_present(
            plotter,
            contact_mesh,
            color="#ef9f2d",
            opacity=0.98,
            ambient=0.45,
            diffuse=0.72,
            specular=0.08,
        )
        plotter.add_mesh(pv.Box(bounds=bounds).outline(), color="black", line_width=0.9)
        plotter.add_axes(line_width=1, labels_off=False)
        plotter.add_text(
            f"{SAMPLE_TITLES[sample]}\n{size_um:g} μm local ROI, void {window['void_fraction']:.1%}",
            position=(0.03, 0.88),
            font_size=10,
            viewport=True,
            color="#222222",
        )
        set_y_up_camera(plotter, bounds)
    out_path = out_dir / f"representative30_local_exposure_{size_um:g}um_majority_r{args.contact_radius}.png"
    plotter.screenshot(str(out_path), transparent_background=True)
    plotter.close()
    print(f"Saved {out_path}")
    return out_path


def write_local_windows(windows: list[dict], args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir_3d)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"representative30_local_exposure_majority_r{args.contact_radius}_windows.csv"
    fieldnames = list(windows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(windows)
    print(f"Saved {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    samples: dict[str, dict] = {}
    for sample in SAMPLES:
        label, meta = load_representative_label(args, sample)
        masks = majority_contact_masks(label, int(args.contact_radius))
        volume_void_fraction = contact_fraction(masks)
        selected_slice = choose_slice(label, masks, volume_void_fraction)
        samples[sample] = {
            "label": label,
            "meta": meta,
            "masks": masks,
            "volume_void_fraction": volume_void_fraction,
            "slice": selected_slice,
        }
        print(
            f"{sample}: volume void-exposed={volume_void_fraction:.4f}, "
            f"slice z={selected_slice['z_index']} void={selected_slice['void_fraction']:.4f}"
        )

    if args.clean_only:
        plot_slice_overlay_clean(samples, args)
        return

    if not args.skip_2d:
        plot_slice_overlay(samples, args)
        write_slice_metrics(samples, args)

    if not args.skip_3d:
        windows = []
        for size_um in args.local_sizes_um:
            for sample in SAMPLES:
                window = select_local_window(sample, samples[sample], float(size_um), args)
                windows.append(window)
                print(
                    f"{sample} {size_um:g} um window: z={window['z0']}:{window['z1']} "
                    f"y={window['y0']}:{window['y1']} x={window['x0']}:{window['x1']} "
                    f"void={window['void_fraction']:.4f}, components={window['large_am_component_count']}"
                )
            render_local_3d(samples, windows, float(size_um), args)
        write_local_windows(windows, args)


if __name__ == "__main__":
    main()

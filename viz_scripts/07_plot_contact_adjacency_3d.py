from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile as tiff


SAMPLES = ("WM", "PFDT")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAM_LABEL = 1
SE_LABEL = 2
VOID_LABEL = 3
CONTACT_SPECS = (
    ("AM-SE contact", SE_LABEL, "#1b9e77"),
    ("AM-Void contact", VOID_LABEL, "#7b3294"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render 3D AM-SE and AM-Void contact adjacency maps.")
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument(
        "--representative-root",
        default=PROJECT_ROOT / "server_results" / "30um-in-plane" / "results",
    )
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "viz" / "contact_adjacency_3d")
    parser.add_argument("--output-name", default="representative30_am_contact_adjacency_3d.png")
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--voxel-size-um", type=float, default=0.07)
    parser.add_argument("--crop-fraction", type=float, default=1.0)
    parser.add_argument("--window-width", type=int, default=2200)
    parser.add_argument("--window-height", type=int, default=1600)
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


def center_crop(array: np.ndarray, fraction: float) -> np.ndarray:
    if fraction >= 0.999:
        return array
    slices = []
    for dim in array.shape:
        n = max(8, int(round(dim * fraction)))
        start = (dim - n) // 2
        slices.append(slice(start, start + n))
    return array[tuple(slices)]


def block_reduce_bool(mask: np.ndarray, factor: int, *, mode: str) -> np.ndarray:
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
    if mode == "any":
        return reshaped.max(axis=(1, 3, 5)).astype(np.float32, copy=False)
    return reshaped.mean(axis=(1, 3, 5), dtype=np.float32)


def six_neighbor(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=bool)
    out[1:, :, :] |= mask[:-1, :, :]
    out[:-1, :, :] |= mask[1:, :, :]
    out[:, 1:, :] |= mask[:, :-1, :]
    out[:, :-1, :] |= mask[:, 1:, :]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


def load_sample_masks(args: argparse.Namespace, sample: str) -> dict:
    field_dir = Path(args.representative_root) / sample / "representative_flux"
    meta = read_json(field_dir / "metadata.json")
    label_zyx = tiff.imread(label_path(Path(args.project_root), sample))
    label_xyz = np.transpose(crop_label_to_metadata(label_zyx, meta), (2, 1, 0))
    label_xyz = center_crop(label_xyz, args.crop_fraction)
    am = label_xyz == CAM_LABEL
    masks = {"am_fraction": block_reduce_bool(am, args.downsample_factor, mode="mean")}
    for title, neighbor_label, _ in CONTACT_SPECS:
        neighbor = label_xyz == neighbor_label
        contact = am & six_neighbor(neighbor)
        masks[title] = block_reduce_bool(contact, args.downsample_factor, mode="any")
    return masks


def make_grid(pv, am_fraction: np.ndarray, contact: np.ndarray, spacing_um: float):
    common = tuple(min(am_fraction.shape[i], contact.shape[i]) for i in range(3))
    slicer = tuple(slice(0, n) for n in common)
    am_fraction = am_fraction[slicer]
    contact = contact[slicer]
    grid = pv.ImageData()
    grid.dimensions = am_fraction.shape
    grid.spacing = (spacing_um, spacing_um, spacing_um)
    grid.point_data["am_fraction"] = am_fraction.ravel(order="F")
    grid.point_data["contact"] = contact.ravel(order="F")
    return grid


def cutaway_grid(pv, grid):
    base = grid.threshold(value=0.05, scalars="am_fraction")
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


def add_contact_panel(plotter, pv, grid, title: str, color: str):
    cut, bounds = cutaway_grid(pv, grid)
    plotter.add_mesh(
        cut,
        color="#b8b8b8",
        opacity=0.28,
        show_scalar_bar=False,
        smooth_shading=False,
        lighting=False,
    )
    contact = cut.threshold(value=0.5, scalars="contact", method="upper", all_scalars=True)
    if contact.n_points:
        plotter.add_mesh(
            contact,
            color=color,
            opacity=0.95,
            show_scalar_bar=False,
            smooth_shading=False,
            lighting=False,
        )
    outline = pv.Box(bounds=bounds).outline()
    plotter.add_mesh(outline, color="black", line_width=0.9)
    plotter.add_axes(line_width=1, labels_off=False)
    plotter.add_text(title, position=(0.03, 0.92), font_size=10, viewport=True)
    return bounds


def set_top_corner_camera(plotter, bounds: tuple[float, float, float, float, float, float]) -> None:
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


def main() -> None:
    args = parse_args()
    pv = require_pyvista()
    pv.OFF_SCREEN = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spacing_um = float(args.voxel_size_um) * int(args.downsample_factor)

    sample_masks = {sample: load_sample_masks(args, sample) for sample in SAMPLES}
    plotter = pv.Plotter(shape=(len(SAMPLES), len(CONTACT_SPECS)), off_screen=True,
                         window_size=(args.window_width, args.window_height))
    plotter.set_background("white")
    first_bounds = None
    for row, sample in enumerate(SAMPLES):
        for col, (contact_title, _, color) in enumerate(CONTACT_SPECS):
            plotter.subplot(row, col)
            grid = make_grid(
                pv,
                sample_masks[sample]["am_fraction"],
                sample_masks[sample][contact_title],
                spacing_um,
            )
            bounds = add_contact_panel(
                plotter,
                pv,
                grid,
                f"{sample} {contact_title}",
                color,
            )
            set_top_corner_camera(plotter, bounds)
            if first_bounds is None:
                first_bounds = bounds

    plotter.link_views()
    if first_bounds is not None:
        set_top_corner_camera(plotter, first_bounds)
    out_path = out_dir / args.output_name
    plotter.screenshot(str(out_path), transparent_background=True)
    plotter.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

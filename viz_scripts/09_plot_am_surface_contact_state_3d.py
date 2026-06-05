from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile as tiff


SAMPLES = ("WM", "PFDT")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAM_LABEL = 1
SE_LABEL = 2
VOID_LABEL = 3
STATE_COLORS = {
    "se": "#72bcd5",
    "void": "#ffd06f",
    "other": "#b8b8b8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an AM surface contact-state map colored by the phase adjacent to each AM surface face."
    )
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument(
        "--representative-root",
        default=PROJECT_ROOT / "server_results" / "30um-in-plane" / "results",
    )
    parser.add_argument("--dataset", choices=["representative30", "whole_roi"], default="representative30")
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "viz" / "am_surface_contact_state_3d")
    parser.add_argument("--voxel-size-um", type=float, default=0.07)
    parser.add_argument("--representative-downsample", type=int, default=6)
    parser.add_argument("--whole-roi-downsample", type=int, default=6)
    parser.add_argument("--crop-fraction", type=float, default=1.0)
    parser.add_argument("--surface-threshold", type=float, default=0.50)
    parser.add_argument("--cutaway-start-fraction", type=float, default=0.56)
    parser.add_argument("--se-opacity", type=float, default=0.82)
    parser.add_argument("--void-opacity", type=float, default=0.98)
    parser.add_argument("--other-opacity", type=float, default=0.0)
    parser.add_argument("--void-state-threshold", type=float, default=0.55)
    parser.add_argument(
        "--max-faces-per-state",
        type=int,
        default=0,
        help="Optional deterministic cap per sample/state for faster previews. 0 renders all faces.",
    )
    parser.add_argument("--window-width", type=int, default=1900)
    parser.add_argument("--window-height", type=int, default=900)
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


def load_label_xyz(args: argparse.Namespace, sample: str) -> np.ndarray:
    label_zyx = tiff.imread(label_path(Path(args.project_root), sample))
    if args.dataset == "representative30":
        meta = read_json(Path(args.representative_root) / sample / "representative_flux" / "metadata.json")
        label_zyx = crop_label_to_metadata(label_zyx, meta)
    label_zyx = center_crop(label_zyx, args.crop_fraction)
    return np.transpose(label_zyx.astype(np.uint8, copy=False), (2, 1, 0))


def block_mode_labels(label_xyz: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return label_xyz.astype(np.uint8, copy=False)
    crop_shape = tuple((dim // factor) * factor for dim in label_xyz.shape)
    cropped = label_xyz[tuple(slice(0, n) for n in crop_shape)]
    reshaped = cropped.reshape(
        crop_shape[0] // factor,
        factor,
        crop_shape[1] // factor,
        factor,
        crop_shape[2] // factor,
        factor,
    )
    counts = np.stack([(reshaped == label).sum(axis=(1, 3, 5)) for label in (0, 1, 2, 3)], axis=0)
    return np.argmax(counts, axis=0).astype(np.uint8, copy=False)


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


def apply_cutaway(label_xyz: np.ndarray, start_fraction: float) -> np.ndarray:
    if start_fraction >= 0.999:
        return label_xyz
    out = label_xyz.copy()
    nx, ny, nz = out.shape
    x0 = int(round(nx * start_fraction))
    y0 = int(round(ny * start_fraction))
    z0 = int(round(nz * start_fraction))
    out[x0:, y0:, z0:] = 0
    return out


def state_name(neighbor_label: int) -> str:
    if neighbor_label == SE_LABEL:
        return "se"
    if neighbor_label == VOID_LABEL:
        return "void"
    return "other"


def face_state_counts(label_xyz: np.ndarray) -> dict[str, int]:
    counts = {"se": 0, "void": 0, "other": 0}
    axes = (
        (0, slice(None, -1), slice(1, None)),
        (0, slice(1, None), slice(None, -1)),
        (1, slice(None, -1), slice(1, None)),
        (1, slice(1, None), slice(None, -1)),
        (2, slice(None, -1), slice(1, None)),
        (2, slice(1, None), slice(None, -1)),
    )
    for axis, am_slice_axis, nb_slice_axis in axes:
        am_slices = [slice(None)] * 3
        nb_slices = [slice(None)] * 3
        am_slices[axis] = am_slice_axis
        nb_slices[axis] = nb_slice_axis
        am = label_xyz[tuple(am_slices)] == CAM_LABEL
        neighbor = label_xyz[tuple(nb_slices)]
        surface = am & (neighbor != CAM_LABEL)
        for label, key in ((SE_LABEL, "se"), (VOID_LABEL, "void")):
            counts[key] += int(np.count_nonzero(surface & (neighbor == label)))
        counts["other"] += int(np.count_nonzero(surface & ~np.isin(neighbor, (SE_LABEL, VOID_LABEL))))
    return counts


def six_neighbor(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=bool)
    out[1:, :, :] |= mask[:-1, :, :]
    out[:-1, :, :] |= mask[1:, :, :]
    out[:, 1:, :] |= mask[:, :-1, :]
    out[:, :-1, :] |= mask[:, 1:, :]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


def make_grid(pv, arrays: dict[str, np.ndarray], spacing_um: float):
    grid = pv.ImageData()
    first = next(iter(arrays.values()))
    grid.dimensions = first.shape
    grid.spacing = (spacing_um, spacing_um, spacing_um)
    for name, array in arrays.items():
        grid.point_data[name] = array.ravel(order="F")
    return grid


def cutaway_polydata(pv, mesh, bounds: tuple[float, float, float, float, float, float], start_fraction: float):
    if start_fraction >= 0.999 or mesh.n_points == 0:
        return mesh
    lx = bounds[1] - bounds[0]
    ly = bounds[3] - bounds[2]
    lz = bounds[5] - bounds[4]
    clip_bounds = (
        bounds[0] + start_fraction * lx,
        bounds[1],
        bounds[2] + start_fraction * ly,
        bounds[3],
        bounds[4] + start_fraction * lz,
        bounds[5],
    )
    return mesh.clip_box(clip_bounds, invert=True)


def as_smooth_surface(mesh):
    if mesh is None or mesh.n_points == 0:
        return mesh
    if not hasattr(mesh, "compute_normals"):
        mesh = mesh.extract_surface(algorithm="dataset_surface")
    return mesh.clean().compute_normals(auto_orient_normals=True, consistent_normals=True)


def make_state_surfaces(
    pv,
    label_xyz: np.ndarray,
    factor: int,
    spacing_um: float,
    surface_threshold: float,
    cutaway_start_fraction: float,
    void_state_threshold: float,
) -> tuple[dict[str, object], tuple[float, float, float, float, float, float]]:
    am = label_xyz == CAM_LABEL
    se_contact = am & six_neighbor(label_xyz == SE_LABEL)
    void_contact = am & six_neighbor(label_xyz == VOID_LABEL)
    am_fraction = block_fraction(am, factor)
    se_fraction = block_fraction(se_contact, factor)
    void_fraction = block_fraction(void_contact, factor)
    classified = se_fraction + void_fraction
    void_state = np.divide(void_fraction, classified, out=np.zeros_like(void_fraction), where=classified > 0)
    grid = make_grid(
        pv,
        {
            "am_fraction": am_fraction,
            "se_contact_fraction": se_fraction,
            "void_contact_fraction": void_fraction,
            "void_state": void_state,
        },
        spacing_um,
    )
    bounds = outline_bounds(am_fraction.shape, spacing_um)
    surface = grid.contour([surface_threshold], scalars="am_fraction")
    surface = cutaway_polydata(pv, surface, bounds, cutaway_start_fraction)
    surface = as_smooth_surface(surface)
    void_surface = surface.threshold(value=void_state_threshold, scalars="void_state", method="upper")
    void_surface = as_smooth_surface(void_surface)
    other_surface = surface.threshold(value=0.01, scalars="se_contact_fraction", method="lower")
    other_surface = as_smooth_surface(other_surface)
    return {"se": surface, "void": void_surface, "other": other_surface}, bounds


def append_faces(
    face_bins: dict[str, list[np.ndarray]],
    state: str,
    fixed_axis: int,
    fixed_coord: np.ndarray,
    u_axis: int,
    u0: np.ndarray,
    v_axis: int,
    v0: np.ndarray,
) -> None:
    n_faces = fixed_coord.size
    if n_faces == 0:
        return
    points = np.zeros((n_faces, 4, 3), dtype=np.float32)
    points[:, :, fixed_axis] = fixed_coord[:, None]
    points[:, 0, u_axis] = u0
    points[:, 1, u_axis] = u0 + 1
    points[:, 2, u_axis] = u0 + 1
    points[:, 3, u_axis] = u0
    points[:, 0, v_axis] = v0
    points[:, 1, v_axis] = v0
    points[:, 2, v_axis] = v0 + 1
    points[:, 3, v_axis] = v0 + 1
    face_bins[state].append(points)


def collect_axis_faces(label_xyz: np.ndarray, axis: int, positive: bool, face_bins: dict[str, list[np.ndarray]]) -> None:
    am_slices = [slice(None)] * 3
    nb_slices = [slice(None)] * 3
    if positive:
        am_slices[axis] = slice(None, -1)
        nb_slices[axis] = slice(1, None)
        face_offset = 1
    else:
        am_slices[axis] = slice(1, None)
        nb_slices[axis] = slice(None, -1)
        face_offset = 0
    am = label_xyz[tuple(am_slices)] == CAM_LABEL
    neighbor = label_xyz[tuple(nb_slices)]
    surface = am & (neighbor != CAM_LABEL)
    if not np.any(surface):
        return

    coords = np.indices(surface.shape, dtype=np.int32)
    for neighbor_label in (SE_LABEL, VOID_LABEL, 0):
        if neighbor_label == 0:
            mask = surface & ~np.isin(neighbor, (SE_LABEL, VOID_LABEL))
        else:
            mask = surface & (neighbor == neighbor_label)
        if not np.any(mask):
            continue
        idx = [coords[dim][mask] for dim in range(3)]
        if not positive:
            idx[axis] = idx[axis] + 1
        fixed_coord = idx[axis] + face_offset
        other_axes = [dim for dim in range(3) if dim != axis]
        state = state_name(neighbor_label)
        append_faces(
            face_bins,
            state,
            axis,
            fixed_coord.astype(np.float32, copy=False),
            other_axes[0],
            idx[other_axes[0]].astype(np.float32, copy=False),
            other_axes[1],
            idx[other_axes[1]].astype(np.float32, copy=False),
        )


def subsample_points(points: np.ndarray, max_faces: int, seed: int) -> np.ndarray:
    if max_faces <= 0 or points.shape[0] <= max_faces:
        return points
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(points.shape[0], size=max_faces, replace=False))
    return points[keep]


def make_polydata(pv, point_blocks: list[np.ndarray], spacing_um: float, max_faces: int, seed: int):
    if not point_blocks:
        return None
    points_by_face = np.concatenate(point_blocks, axis=0)
    points_by_face = subsample_points(points_by_face, max_faces, seed)
    n_faces = points_by_face.shape[0]
    points = (points_by_face.reshape(-1, 3) * spacing_um).astype(np.float32, copy=False)
    faces = np.empty((n_faces, 5), dtype=np.int64)
    faces[:, 0] = 4
    faces[:, 1:] = np.arange(n_faces * 4, dtype=np.int64).reshape(n_faces, 4)
    return pv.PolyData(points, faces.ravel())


def make_state_meshes(pv, label_xyz: np.ndarray, spacing_um: float, max_faces: int) -> dict[str, object]:
    face_bins: dict[str, list[np.ndarray]] = {"se": [], "void": [], "other": []}
    for axis in range(3):
        collect_axis_faces(label_xyz, axis, True, face_bins)
        collect_axis_faces(label_xyz, axis, False, face_bins)
    return {
        state: make_polydata(pv, blocks, spacing_um, max_faces, seed=i + 17)
        for i, (state, blocks) in enumerate(face_bins.items())
    }


def outline_bounds(label_shape: tuple[int, int, int], spacing_um: float) -> tuple[float, float, float, float, float, float]:
    return (
        0.0,
        label_shape[0] * spacing_um,
        0.0,
        label_shape[1] * spacing_um,
        0.0,
        label_shape[2] * spacing_um,
    )


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
    direction = np.array([1.0, 0.80, 1.0], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    plotter.camera.position = tuple(center + distance * direction)
    plotter.camera.focal_point = tuple(center)
    plotter.camera.view_up = (0.0, 1.0, 0.0)
    plotter.camera.zoom(1.04)


def add_lighting(plotter, pv) -> None:
    try:
        plotter.enable_anti_aliasing("ssaa")
        plotter.enable_eye_dome_lighting()
    except Exception:
        pass
    try:
        plotter.add_light(pv.Light(position=(1.0, 1.6, 1.8), focal_point=(0, 0, 0), intensity=0.75))
        plotter.add_light(pv.Light(position=(-1.2, 0.8, 0.4), focal_point=(0, 0, 0), intensity=0.35))
    except Exception:
        pass


def add_contact_state_panel(
    plotter,
    pv,
    sample: str,
    meshes: dict[str, object],
    bounds: tuple[float, float, float, float, float, float],
    metrics: dict[str, float],
    se_opacity: float,
    void_opacity: float,
    other_opacity: float,
) -> None:
    order = (
        ("other", other_opacity),
        ("se", se_opacity),
        ("void", void_opacity),
    )
    for state, opacity in order:
        mesh = meshes.get(state)
        if mesh is None or mesh.n_points == 0 or opacity <= 0:
            continue
        plotter.add_mesh(
            mesh,
            color=STATE_COLORS[state],
            opacity=opacity,
            show_scalar_bar=False,
            smooth_shading=False,
            lighting=True,
            ambient=0.20,
            diffuse=0.82,
            specular=0.08,
        )
    outline = pv.Box(bounds=bounds).outline()
    plotter.add_mesh(outline, color="black", line_width=0.9)
    plotter.add_axes(line_width=1, labels_off=False)
    title = (
        f"{sample}\n"
        f"SE-covered AM surface {metrics['se_fraction']:.1%}\n"
        f"Void-exposed AM surface {metrics['void_fraction']:.1%}"
    )
    plotter.add_text(title, position=(0.03, 0.88), font_size=10, viewport=True, color="#222222")
    set_y_up_camera(plotter, bounds)


def add_legend(plotter) -> None:
    lines = (
        ("SE-rich contact", "se"),
        ("Void/carbon contact", "void"),
        ("Other / cut", "other"),
    )
    y = 0.18
    for label, state in lines:
        plotter.add_text(label, position=(0.75, y), font_size=9, viewport=True, color=STATE_COLORS[state])
        y -= 0.045


def write_metrics(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_path}")


def main() -> None:
    args = parse_args()
    pv = require_pyvista()
    pv.OFF_SCREEN = True
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    factor = args.representative_downsample if args.dataset == "representative30" else args.whole_roi_downsample
    spacing_um = float(args.voxel_size_um) * int(factor)
    render_data = {}
    metric_rows = []
    for sample in SAMPLES:
        label_xyz = load_label_xyz(args, sample)
        counts = face_state_counts(label_xyz)
        contact_total = counts["se"] + counts["void"]
        metric_rows.append(
            {
                "dataset": args.dataset,
                "sample": sample,
                "am_se_face_count": counts["se"],
                "am_void_face_count": counts["void"],
                "am_other_face_count": counts["other"],
                "am_se_fraction_of_classified_contact": counts["se"] / contact_total if contact_total else np.nan,
                "am_void_fraction_of_classified_contact": counts["void"] / contact_total if contact_total else np.nan,
            }
        )
        meshes, bounds = make_state_surfaces(
            pv,
            label_xyz,
            int(factor),
            spacing_um,
            args.surface_threshold,
            args.cutaway_start_fraction,
            args.void_state_threshold,
        )
        render_data[sample] = {
            "meshes": meshes,
            "bounds": bounds,
            "metrics": {
                "se_fraction": metric_rows[-1]["am_se_fraction_of_classified_contact"],
                "void_fraction": metric_rows[-1]["am_void_fraction_of_classified_contact"],
            },
        }
        face_summary = ", ".join(
            f"{state}={mesh.n_cells if mesh is not None else 0}" for state, mesh in meshes.items()
        )
        print(f"{args.dataset} {sample}: rendered surface cells {face_summary}")

    plotter = pv.Plotter(shape=(1, len(SAMPLES)), off_screen=True, window_size=(args.window_width, args.window_height))
    plotter.set_background("white")
    add_lighting(plotter, pv)
    first_bounds = None
    for col, sample in enumerate(SAMPLES):
        plotter.subplot(0, col)
        add_contact_state_panel(
            plotter,
            pv,
            sample,
            render_data[sample]["meshes"],
            render_data[sample]["bounds"],
            render_data[sample]["metrics"],
            args.se_opacity,
            args.void_opacity,
            args.other_opacity,
        )
        if first_bounds is None:
            first_bounds = render_data[sample]["bounds"]
    plotter.link_views()
    if first_bounds is not None:
        set_y_up_camera(plotter, first_bounds)
    add_legend(plotter)

    out_path = out_dir / f"{args.dataset}_am_surface_contact_state_3d.png"
    plotter.screenshot(str(out_path), transparent_background=True)
    plotter.close()
    print(f"Saved {out_path}")
    write_metrics(metric_rows, out_dir / f"{args.dataset}_am_surface_contact_state_metrics.csv")


if __name__ == "__main__":
    main()

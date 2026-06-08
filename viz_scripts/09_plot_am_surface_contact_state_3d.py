from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile as tiff


SAMPLES = ("WM", "PFDT")
SAMPLE_TITLES = {
    "WM": "WM-LPSCB",
    "PFDT": "PFDT@LPSCB",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAM_LABEL = 1
SE_LABEL = 2
VOID_LABEL = 3
STATE_COLORS = {
    "base": "#c8c8c8",
    "se": "#72bcd5",
    "void": "#f2b84b",
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
    parser.add_argument(
        "--contact-radius",
        type=int,
        default=2,
        help="Voxel radius used to assign each AM surface voxel by local SE/void majority.",
    )
    parser.add_argument(
        "--contact-mode",
        choices=["majority", "face1"],
        default="majority",
        help="Contact-state definition used for the rendered surface. Metrics CSV always includes both modes.",
    )
    parser.add_argument("--cutaway-start-fraction", type=float, default=0.56)
    parser.add_argument("--base-opacity", type=float, default=0.12)
    parser.add_argument("--se-opacity", type=float, default=0.0)
    parser.add_argument("--void-opacity", type=float, default=0.98)
    parser.add_argument("--other-opacity", type=float, default=0.0)
    parser.add_argument("--void-state-threshold", type=float, default=0.30)
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


def contact_total(counts: dict[str, int]) -> int:
    return int(counts.get("se", 0) + counts.get("void", 0))


def counts_to_metric_row(
    dataset: str,
    sample: str,
    mode: str,
    counts: dict[str, int],
    unit: str,
    radius: int | None = None,
) -> dict[str, int | float | str]:
    total = contact_total(counts)
    surface_total = int(counts.get("surface", counts.get("se", 0) + counts.get("void", 0) + counts.get("other", 0)))
    return {
        "dataset": dataset,
        "sample": sample,
        "contact_mode": mode,
        "contact_unit": unit,
        "contact_radius_voxels": "" if radius is None else radius,
        "am_surface_total_count": surface_total,
        "am_se_count": counts.get("se", 0),
        "am_void_count": counts.get("void", 0),
        "am_other_count": counts.get("other", 0),
        "am_se_fraction_of_classified_contact": counts.get("se", 0) / total if total else np.nan,
        "am_void_fraction_of_classified_contact": counts.get("void", 0) / total if total else np.nan,
        "am_se_fraction_of_am_surface": counts.get("se", 0) / surface_total if surface_total else np.nan,
        "am_void_fraction_of_am_surface": counts.get("void", 0) / surface_total if surface_total else np.nan,
    }


def six_neighbor(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=bool)
    out[1:, :, :] |= mask[:-1, :, :]
    out[:-1, :, :] |= mask[1:, :, :]
    out[:, 1:, :] |= mask[:, :-1, :]
    out[:, :-1, :] |= mask[:, 1:, :]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


def am_surface_mask(label_xyz: np.ndarray) -> np.ndarray:
    am = label_xyz == CAM_LABEL
    return am & six_neighbor(~am)


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
        for dx in range(2 * radius + 1):
            for dy in range(2 * radius + 1):
                for dz in range(2 * radius + 1):
                    out += padded[
                        dx : dx + mask.shape[0],
                        dy : dy + mask.shape[1],
                        dz : dz + mask.shape[2],
                    ]
        return out


def majority_contact_masks(label_xyz: np.ndarray, radius: int) -> dict[str, np.ndarray]:
    surface = am_surface_mask(label_xyz)
    se_count = local_phase_count(label_xyz == SE_LABEL, radius)
    void_count = local_phase_count(label_xyz == VOID_LABEL, radius)
    classified = surface & ((se_count + void_count) > 0)
    void_exposed = classified & (void_count > se_count)
    se_covered = classified & ~void_exposed
    other = surface & ~classified
    return {"surface": surface, "se": se_covered, "void": void_exposed, "other": other}


def face_contact_masks(label_xyz: np.ndarray) -> dict[str, np.ndarray]:
    surface = am_surface_mask(label_xyz)
    void_exposed = surface & six_neighbor(label_xyz == VOID_LABEL)
    se_covered = surface & six_neighbor(label_xyz == SE_LABEL) & ~void_exposed
    other = surface & ~(se_covered | void_exposed)
    return {"surface": surface, "se": se_covered, "void": void_exposed, "other": other}


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
    contact_masks: dict[str, np.ndarray],
    factor: int,
    spacing_um: float,
    surface_threshold: float,
    cutaway_start_fraction: float,
    void_state_threshold: float,
) -> tuple[dict[str, object], tuple[float, float, float, float, float, float]]:
    am = label_xyz == CAM_LABEL
    am_fraction = block_fraction(am, factor)
    surface_fraction = block_fraction(contact_masks["surface"], factor)
    se_fraction = block_fraction(contact_masks["se"], factor)
    void_fraction = block_fraction(contact_masks["void"], factor)
    other_fraction = block_fraction(contact_masks["other"], factor)
    void_state = np.divide(void_fraction, surface_fraction, out=np.zeros_like(void_fraction), where=surface_fraction > 0)
    se_state = np.divide(se_fraction, surface_fraction, out=np.zeros_like(se_fraction), where=surface_fraction > 0)
    grid = make_grid(
        pv,
        {
            "am_fraction": am_fraction,
            "surface_fraction": surface_fraction,
            "se_state": se_state,
            "void_state": void_state,
            "other_fraction": other_fraction,
        },
        spacing_um,
    )
    bounds = outline_bounds(am_fraction.shape, spacing_um)
    surface = grid.contour([surface_threshold], scalars="am_fraction")
    surface = cutaway_polydata(pv, surface, bounds, cutaway_start_fraction)
    surface = as_smooth_surface(surface)
    void_surface = surface.threshold(value=void_state_threshold, scalars="void_state", method="upper")
    void_surface = as_smooth_surface(void_surface)
    se_surface = surface.threshold(value=0.50, scalars="se_state", method="upper")
    se_surface = as_smooth_surface(se_surface)
    other_surface = surface.threshold(value=0.01, scalars="other_fraction", method="upper")
    other_surface = as_smooth_surface(other_surface)
    return {"base": surface, "se": se_surface, "void": void_surface, "other": other_surface}, bounds


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
    base_opacity: float,
    se_opacity: float,
    void_opacity: float,
    other_opacity: float,
) -> None:
    order = (
        ("base", base_opacity),
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
            ambient=0.46 if state == "void" else 0.30,
            diffuse=0.72,
            specular=0.04,
        )
    outline = pv.Box(bounds=bounds).outline()
    plotter.add_mesh(outline, color="black", line_width=0.9)
    plotter.add_axes(line_width=1, labels_off=False)
    title = (
        f"{SAMPLE_TITLES.get(sample, sample)}\n"
        f"Void-exposed AM surface {metrics['void_fraction']:.1%}"
    )
    plotter.add_text(title, position=(0.03, 0.88), font_size=10, viewport=True, color="#222222")
    set_y_up_camera(plotter, bounds)


def add_legend(plotter) -> None:
    lines = (
        ("AM surface", "base"),
        ("Void/carbon-exposed AM", "void"),
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


def render_mode_token(args: argparse.Namespace) -> str:
    if args.contact_mode == "majority":
        return f"majority_r{args.contact_radius}"
    return "face1"


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
        face_counts = face_state_counts(label_xyz)
        majority_masks = majority_contact_masks(label_xyz, args.contact_radius)
        majority_counts = {key: int(np.count_nonzero(mask)) for key, mask in majority_masks.items()}
        metric_rows.append(counts_to_metric_row(args.dataset, sample, "face1", face_counts, "voxel_face"))
        metric_rows.append(
            counts_to_metric_row(
                args.dataset,
                sample,
                f"majority_r{args.contact_radius}",
                majority_counts,
                "surface_voxel",
                args.contact_radius,
            )
        )
        contact_masks = majority_masks if args.contact_mode == "majority" else face_contact_masks(label_xyz)
        render_counts = majority_counts if args.contact_mode == "majority" else {
            key: int(np.count_nonzero(mask)) for key, mask in contact_masks.items()
        }
        render_row = counts_to_metric_row(
            args.dataset,
            sample,
            args.contact_mode,
            render_counts,
            "surface_voxel",
            args.contact_radius if args.contact_mode == "majority" else None,
        )
        meshes, bounds = make_state_surfaces(
            pv,
            label_xyz,
            contact_masks,
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
                "se_fraction": render_row["am_se_fraction_of_classified_contact"],
                "void_fraction": render_row["am_void_fraction_of_classified_contact"],
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
            args.base_opacity,
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

    mode_token = render_mode_token(args)
    out_path = out_dir / f"{args.dataset}_am_surface_void_exposed_{mode_token}_3d.png"
    plotter.screenshot(str(out_path), transparent_background=True)
    plotter.close()
    print(f"Saved {out_path}")
    write_metrics(metric_rows, out_dir / f"{args.dataset}_am_surface_contact_definition_comparison.csv")


if __name__ == "__main__":
    main()

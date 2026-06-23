from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

from viz_style import apply_publication_style, panel_figsize, save_figure, style_axes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ("WM", "PFDT")
PHASES = (
    ("CAM", 1, "#376795"),
    ("SE-rich", 2, "#72bcd5"),
    ("Void/C-rich", 3, "#ffd06f"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render visible internal void distribution and whole-ROI phase-fraction profiles."
    )
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument(
        "--representative-root",
        default=PROJECT_ROOT / "server_results" / "30um-in-plane" / "results",
    )
    parser.add_argument("--output-dir-3d", default=PROJECT_ROOT / "viz" / "representative30_3d_cutaway")
    parser.add_argument("--output-dir-profile", default=PROJECT_ROOT / "viz" / "geometry_metrics")
    parser.add_argument("--voxel-size-um", type=float, default=0.07)
    parser.add_argument("--representative-downsample", type=int, default=2)
    parser.add_argument("--whole-roi-bin-um", type=float, default=2.0)
    parser.add_argument("--matrix-color", default="#eee8c9")
    parser.add_argument("--matrix-opacity", type=float, default=0.18)
    parser.add_argument("--void-color", default="#3569b1")
    parser.add_argument("--void-opacity", type=float, default=0.82)
    parser.add_argument("--void-threshold", type=float, default=0.50)
    parser.add_argument("--matrix-threshold", type=float, default=0.45)
    parser.add_argument("--window-width", type=int, default=2200)
    parser.add_argument("--window-height", type=int, default=1200)
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    return parser.parse_args()


def require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise SystemExit("PyVista is required for the 3D void render: pip install pyvista vtk") from exc
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


def make_grid(pv, arrays: dict[str, np.ndarray], spacing_um: float):
    shape = next(iter(arrays.values())).shape
    grid = pv.ImageData()
    grid.dimensions = tuple(dim + 1 for dim in shape)
    grid.spacing = (spacing_um, spacing_um, spacing_um)
    for name, array in arrays.items():
        grid.cell_data[name] = array.ravel(order="F")
    return grid


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


def representative_label_xyz(args: argparse.Namespace, sample: str) -> np.ndarray:
    field_dir = Path(args.representative_root) / sample / "representative_flux"
    meta = read_json(field_dir / "metadata.json")
    label_zyx = tiff.imread(label_path(Path(args.project_root), sample)).astype(np.uint8, copy=False)
    return zyx_to_xyz(crop_label_to_metadata(label_zyx, meta))


def render_void_distribution(args: argparse.Namespace) -> Path:
    pv = require_pyvista()
    pv.OFF_SCREEN = True
    out_dir = Path(args.output_dir_3d)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = {}
    for sample in SAMPLES:
        label_xyz = representative_label_xyz(args, sample)
        matrix = block_fraction(np.isin(label_xyz, (1, 2)), int(args.representative_downsample))
        void = block_fraction(label_xyz == 3, int(args.representative_downsample))
        spacing_um = float(args.voxel_size_um) * int(args.representative_downsample)
        entries[sample] = {
            "grid": make_grid(pv, {"matrix": matrix, "void": void}, spacing_um),
            "void_fraction": float(np.count_nonzero(label_xyz == 3) / label_xyz.size),
            "shape": matrix.shape,
        }
        print(
            f"representative30 {sample}: downsampled={matrix.shape}, "
            f"void/C-rich={100 * entries[sample]['void_fraction']:.2f}%"
        )

    plotter = pv.Plotter(shape=(1, len(SAMPLES)), off_screen=True, window_size=(args.window_width, args.window_height))
    plotter.set_background("white")
    try:
        plotter.enable_lightkit()
    except Exception:
        pass
    first_bounds = None
    for col, sample in enumerate(SAMPLES):
        plotter.subplot(0, col)
        grid = entries[sample]["grid"]
        matrix = grid.threshold(value=float(args.matrix_threshold), scalars="matrix")
        bounds = matrix.bounds
        void = grid.threshold(value=float(args.void_threshold), scalars="void")
        plotter.add_mesh(
            matrix,
            color=args.matrix_color,
            opacity=float(args.matrix_opacity),
            show_scalar_bar=False,
            smooth_shading=False,
            lighting=True,
            ambient=0.42,
            diffuse=0.70,
            specular=0.08,
        )
        if void.n_points:
            plotter.add_mesh(
                void,
                color=args.void_color,
                opacity=float(args.void_opacity),
                show_scalar_bar=False,
                smooth_shading=True,
                lighting=True,
                ambient=0.26,
                diffuse=0.82,
                specular=0.24,
                specular_power=18,
            )
        outline = pv.Box(bounds=bounds).outline()
        plotter.add_mesh(outline, color="black", line_width=0.9)
        plotter.add_axes(line_width=1, labels_off=False)
        plotter.add_text(
            f"{sample} void/C-rich {100 * entries[sample]['void_fraction']:.1f}%",
            position=(0.03, 0.93),
            font_size=10,
            viewport=True,
        )
        set_y_up_camera(plotter, bounds)
        if first_bounds is None:
            first_bounds = bounds
    plotter.link_views()
    if first_bounds is not None:
        set_y_up_camera(plotter, first_bounds)
    out_path = out_dir / "representative30_void_distribution_transparent_matrix.png"
    plotter.screenshot(str(out_path), transparent_background=True)
    plotter.close()
    print(f"Saved {out_path}")
    return out_path


def phase_profile_for_label(label_zyx: np.ndarray, voxel_size_um: float, bin_um: float) -> dict[str, np.ndarray]:
    y_count = label_zyx.shape[1]
    y_length_um = y_count * voxel_size_um
    bin_voxels = max(1, int(round(bin_um / voxel_size_um)))
    rows = []
    distances = []
    for y0 in range(0, y_count, bin_voxels):
        y1 = min(y_count, y0 + bin_voxels)
        slab = label_zyx[:, y0:y1, :]
        total = slab.size
        rows.append([100.0 * np.count_nonzero(slab == label) / total for _, label, _ in PHASES])
        old_y_center_um = (0.5 * (y0 + y1)) * voxel_size_um
        distances.append(y_length_um - old_y_center_um)
    fractions = np.asarray(rows, dtype=np.float32)
    distance_array = np.asarray(distances, dtype=np.float32)
    order = np.argsort(distance_array)
    return {"distance_um": distance_array[order], "fractions": fractions[order]}


def render_whole_roi_phase_profile(args: argparse.Namespace) -> list[Path]:
    apply_publication_style()
    out_dir = Path(args.output_dir_profile)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = {}
    for sample in SAMPLES:
        label_zyx = tiff.memmap(label_path(Path(args.project_root), sample))
        profiles[sample] = phase_profile_for_label(label_zyx, float(args.voxel_size_um), float(args.whole_roi_bin_um))
        print(f"whole ROI {sample}: profile bins={profiles[sample]['fractions'].shape[0]}")

    out_paths = []
    for sample in SAMPLES:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=panel_figsize(ncols=1, panel_width_mm=66.80, panel_height_mm=53.97, extra_width_mm=13),
            constrained_layout=True,
        )
        distance = profiles[sample]["distance_um"]
        fractions = profiles[sample]["fractions"]
        width = float(args.whole_roi_bin_um) * 0.92
        bottom = np.zeros_like(distance, dtype=np.float32)
        for i, (name, _, color) in enumerate(PHASES):
            ax.bar(
                distance,
                fractions[:, i],
                width=width,
                bottom=bottom,
                color=color,
                edgecolor="black",
                linewidth=0.25,
                label=name,
                align="center",
            )
            bottom += fractions[:, i]
        ax.set_title(sample)
        ax.set_xlabel("Distance from current collector (μm)")
        ax.set_xlim(0, max(distance) + 0.5 * float(args.whole_roi_bin_um))
        ax.set_ylim(0, 100)
        style_axes(ax)
        ax.set_ylabel("Volume fraction (%)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.2)
        out_path = out_dir / f"whole_roi_phase_fraction_profile_from_current_collector_{sample}.png"
        save_figure(fig, out_path)
        plt.close(fig)
        out_paths.append(out_path)
        print(f"Saved {out_path}")
    return out_paths


def main() -> None:
    args = parse_args()
    outputs = []
    if not args.skip_3d:
        outputs.append(render_void_distribution(args))
    if not args.skip_profile:
        outputs.extend(render_whole_roi_phase_profile(args))
    print("\nRendered files:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()

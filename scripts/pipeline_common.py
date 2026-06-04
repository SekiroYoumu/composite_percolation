from __future__ import annotations

import csv
import inspect
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff


AXES = ("x", "y", "z")
ZYX = ("z", "y", "x")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | os.PathLike[str] = "config.json") -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = repo_root() / path
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(path)
    cfg["_root"] = str(path.parent)
    return cfg


def root_path(cfg: dict[str, Any], *parts: str) -> Path:
    return Path(cfg["_root"]).joinpath(*parts)


def input_path(cfg: dict[str, Any], filename: str) -> Path:
    return root_path(cfg, cfg.get("input_dir", "input"), filename)


def sample_results_dir(cfg: dict[str, Any], sample_key: str) -> Path:
    path = root_path(cfg, cfg.get("results_dir", "results"), sample_key)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sample_label_path(cfg: dict[str, Any], sample_key: str, prefer_existing: bool = True) -> Path:
    """Path for the merged integer label volume.

    Labels are intermediate files used by PuMA. New runs store them outside
    results/ so that downloadable outputs stay lightweight. Existing projects
    that still have results/<sample>/label_zyx.tif remain readable.
    """
    label_dir = cfg.get("label_dir", "intermediate/labels")
    new_path = root_path(cfg, label_dir, sample_key, "label_zyx.tif")
    old_path = root_path(cfg, cfg.get("results_dir", "results"), sample_key, "label_zyx.tif")
    if prefer_existing:
        if new_path.exists():
            return new_path
        if old_path.exists():
            return old_path
    return new_path


def sample_fields_dir(cfg: dict[str, Any], sample_key: str) -> Path:
    storage_dir = cfg.get("field_storage_dir")
    if storage_dir:
        path = root_path(cfg, storage_dir, sample_key, "fields")
    else:
        path = sample_results_dir(cfg, sample_key) / "fields"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(cfg: dict[str, Any]) -> Path:
    path = root_path(cfg, cfg.get("results_dir", "results"), "figures")
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_axis_order(axis_order: str) -> tuple[str, str, str]:
    axes = tuple(axis_order.lower().split("_"))
    if sorted(axes) != ["x", "y", "z"]:
        raise ValueError(f"Invalid axis order {axis_order!r}; expected a permutation like z_y_x.")
    return axes  # type: ignore[return-value]


def transpose_between_orders(array: np.ndarray, from_order: str, to_order: str) -> np.ndarray:
    src = parse_axis_order(from_order)
    dst = parse_axis_order(to_order)
    perm = [src.index(axis) for axis in dst]
    return np.transpose(array, perm)


def read_volume_as_zyx(path: Path, tiff_axis_order: str = "z_y_x") -> np.ndarray:
    array = tiff.imread(path)
    if array.ndim != 3:
        raise ValueError(f"{path} is {array.ndim}D; expected a 3D TIFF stack.")
    if tiff_axis_order != "z_y_x":
        array = transpose_between_orders(array, tiff_axis_order, "z_y_x")
    return array


def write_label_tiff(path: Path, label_zyx: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(path, label_zyx.astype(np.uint8, copy=False), bigtiff=True)


def compact_unique(array: np.ndarray, max_values: int = 16) -> str:
    values, counts = np.unique(array, return_counts=True)
    pairs = [f"{v}:{c}" for v, c in zip(values[:max_values], counts[:max_values])]
    if len(values) > max_values:
        pairs.append(f"... total_unique={len(values)}")
    return "{" + ", ".join(pairs) + "}"


def voxel_size_by_axis(cfg: dict[str, Any]) -> dict[str, float]:
    value = cfg["voxel_size_um"]
    if isinstance(value, (int, float)):
        return {"x": float(value), "y": float(value), "z": float(value)}
    if isinstance(value, list):
        if len(value) != 3:
            raise ValueError("voxel_size_um list must be [x, y, z].")
        return {"x": float(value[0]), "y": float(value[1]), "z": float(value[2])}
    if isinstance(value, dict):
        return {axis: float(value[axis]) for axis in AXES}
    raise TypeError("voxel_size_um must be a scalar, [x, y, z], or {x, y, z}.")


def um_size_to_zyx_voxels(size_um: Iterable[float], cfg: dict[str, Any]) -> tuple[int, int, int]:
    size = list(size_um)
    if len(size) != 3:
        raise ValueError("Physical size must be [x_um, y_um, z_um].")
    voxel = voxel_size_by_axis(cfg)
    x = max(1, int(round(float(size[0]) / voxel["x"])))
    y = max(1, int(round(float(size[1]) / voxel["y"])))
    z = max(1, int(round(float(size[2]) / voxel["z"])))
    return z, y, x


def center_crop_slices(shape_zyx: tuple[int, int, int], size_zyx: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    slices = []
    for dim, requested in zip(shape_zyx, size_zyx):
        n = min(dim, requested)
        start = (dim - n) // 2
        slices.append(slice(start, start + n))
    return tuple(slices)  # type: ignore[return-value]


def label_fractions(label_zyx: np.ndarray, labels: dict[str, int]) -> dict[str, float]:
    total = int(label_zyx.size)
    out: dict[str, float] = {}
    for phase, label in labels.items():
        out[f"{phase} fraction"] = float(np.count_nonzero(label_zyx == int(label)) / total)
    out["unassigned fraction"] = float(np.count_nonzero(label_zyx == 0) / total)
    return out


def label_counts(label_zyx: np.ndarray, labels: dict[str, int]) -> dict[str, int]:
    out = {phase: int(np.count_nonzero(label_zyx == int(label))) for phase, label in labels.items()}
    out["unassigned"] = int(np.count_nonzero(label_zyx == 0))
    return out


def save_label_preview(label_zyx: np.ndarray, path: Path, labels: dict[str, int], title: str) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    path.parent.mkdir(parents=True, exist_ok=True)
    z_indices = sorted(set(int(round(v)) for v in np.linspace(0, label_zyx.shape[0] - 1, 3)))
    cmap = ListedColormap(["black", "#d95f02", "#1b9e77", "#7570b3"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, axes = plt.subplots(1, len(z_indices), figsize=(5 * len(z_indices), 5), constrained_layout=True)
    if len(z_indices) == 1:
        axes = [axes]
    for ax, z in zip(axes, z_indices):
        ax.imshow(label_zyx[z], cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"{title}: z={z}")
        ax.set_axis_off()
    handles = [
        plt.Line2D([0], [0], color="black", lw=8, label="0 unassigned"),
        plt.Line2D([0], [0], color="#d95f02", lw=8, label=f"{labels['CAM']} CAM"),
        plt.Line2D([0], [0], color="#1b9e77", lw=8, label=f"{labels['SE-rich']} SE-rich"),
        plt.Line2D([0], [0], color="#7570b3", lw=8, label=f"{labels['Void/carbon-rich']} Void/carbon-rich"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def zyx_to_puma_array(label_zyx: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    puma_axis_order = cfg.get("puma_axis_order", "x_y_z")
    return transpose_between_orders(label_zyx, "z_y_x", puma_axis_order)


def normalize_side_bc(side_bc: str) -> str:
    mapping = {
        "symmetric": "s",
        "sym": "s",
        "s": "s",
        "periodic": "p",
        "p": "p",
        "dirichlet": "d",
        "d": "d",
    }
    return mapping.get(str(side_bc).lower(), side_bc)


def available_pumapy_functions() -> list[str]:
    try:
        import pumapy as puma
    except Exception as exc:
        return [f"Could not import pumapy: {exc}"]
    names = [name for name in dir(puma) if not name.startswith("_")]
    return sorted(name for name in names if "conduct" in name.lower() or "workspace" in name.lower())


def _make_puma_workspace(array: np.ndarray, cfg: dict[str, Any]) -> Any:
    import pumapy as puma

    if hasattr(puma, "Workspace") and hasattr(puma.Workspace, "from_array"):
        ws = puma.Workspace.from_array(array)
    elif hasattr(puma, "Workspace"):
        try:
            ws = puma.Workspace(array)
        except TypeError:
            ws = puma.Workspace()
            if hasattr(ws, "matrix"):
                ws.matrix = array
            elif hasattr(ws, "set_matrix"):
                ws.set_matrix(array)
            else:
                raise
    else:
        raise RuntimeError("pumapy.Workspace is not available in this PuMA installation.")

    voxel = voxel_size_by_axis(cfg)
    if len(set(voxel.values())) == 1:
        voxel_length = next(iter(voxel.values()))
        if hasattr(ws, "voxel_length"):
            ws.voxel_length = voxel_length
        elif hasattr(ws, "set_voxel_length"):
            ws.set_voxel_length(voxel_length)
    return ws


def _make_conductivity_map(conductivity: dict[str, float]) -> Any:
    import pumapy as puma

    map_classes = ["IsotropicConductivityMap", "ConductivityMap", "MaterialPropertyMap"]
    cond_map = None
    for cls_name in map_classes:
        cls = getattr(puma, cls_name, None)
        if cls is None:
            continue
        try:
            cond_map = cls()
            break
        except TypeError:
            continue
    if cond_map is None:
        return {int(k): float(v) for k, v in conductivity.items()}

    for label, value in conductivity.items():
        label_i = int(label)
        value_f = float(value)
        if hasattr(cond_map, "add_material"):
            try:
                cond_map.add_material((label_i, label_i), value_f)
            except TypeError:
                cond_map.add_material(label_i, value_f)
        elif hasattr(cond_map, "add_isotropic_material"):
            cond_map.add_isotropic_material((label_i, label_i), value_f)
        else:
            raise RuntimeError(f"{type(cond_map).__name__} has no known material-add method.")
    return cond_map


def _call_with_supported_kwargs(func: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        sig = inspect.signature(func)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if accepts_kwargs:
            filtered = kwargs
        else:
            filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    except (TypeError, ValueError):
        filtered = kwargs
    return func(*args, **filtered)


def _parse_transport_result(result: Any, direction: str) -> dict[str, Any]:
    if isinstance(result, dict):
        keff = result.get("keff", result.get("effective_conductivity", result.get("k_eff")))
        potential = result.get("potential", result.get("temperature", result.get("field")))
        flux = result.get("flux", result.get("q"))
    elif isinstance(result, tuple):
        keff = result[0] if len(result) > 0 else None
        potential = result[1] if len(result) > 1 else None
        flux = result[2] if len(result) > 2 else None
    else:
        keff = result
        potential = None
        flux = None

    return {
        "keff_norm": extract_directional_scalar(keff, direction),
        "keff_raw": keff,
        "potential": potential,
        "flux": flux,
        "flux_magnitude": flux_magnitude(flux),
    }


def extract_directional_scalar(value: Any, direction: str) -> float:
    if value is None:
        return float("nan")
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    axis_index = {"x": 0, "y": 1, "z": 2}[direction]
    if arr.ndim == 1 and arr.size >= 3:
        return float(arr[axis_index])
    if arr.ndim >= 2 and arr.shape[0] >= 3 and arr.shape[1] >= 3:
        return float(arr[axis_index, axis_index])
    return float(np.ravel(arr)[0])


def flux_magnitude(flux: Any) -> np.ndarray | None:
    if flux is None:
        return None
    if isinstance(flux, (list, tuple)) and len(flux) == 3:
        comps = [np.asarray(c, dtype=float) for c in flux]
        return np.sqrt(sum(c * c for c in comps))
    arr = np.asarray(flux, dtype=float)
    if arr.ndim >= 1 and arr.shape[-1] == 3:
        return np.linalg.norm(arr, axis=-1)
    if arr.ndim >= 1 and arr.shape[0] == 3:
        return np.sqrt(np.sum(arr * arr, axis=0))
    return np.abs(arr)


def compute_transport(label_zyx: np.ndarray, cfg: dict[str, Any], prefer_electrical: bool = True) -> dict[str, Any]:
    import pumapy as puma

    direction = cfg["through_plane_axis"].lower()
    if direction not in AXES:
        raise ValueError("through_plane_axis must be x, y, or z.")

    ws = _make_puma_workspace(zyx_to_puma_array(label_zyx, cfg), cfg)
    cond_map = _make_conductivity_map(cfg["conductivity"])
    solver_cfg = cfg.get("solver", {})
    kwargs = {
        "direction": direction,
        "side_bc": normalize_side_bc(cfg.get("side_bc", "symmetric")),
        "solver_type": solver_cfg.get("solver_type", "cg"),
        "tolerance": solver_cfg.get("tolerance", 1e-5),
        "maxiter": solver_cfg.get("maxiter", 10000),
        "display_iter": solver_cfg.get("display_iter", False),
    }

    candidates = []
    if prefer_electrical and hasattr(puma, "compute_electrical_conductivity"):
        candidates.append(("electrical", puma.compute_electrical_conductivity))
    if hasattr(puma, "compute_thermal_conductivity"):
        candidates.append(("thermal_fallback", puma.compute_thermal_conductivity))
    if not candidates:
        raise RuntimeError(
            "No supported PuMA conductivity function found. Available conductivity-related functions: "
            + ", ".join(available_pumapy_functions())
        )

    errors = []
    for solver_name, func in candidates:
        try:
            start = time.perf_counter()
            result = _call_with_supported_kwargs(func, ws, cond_map, **kwargs)
            parsed = _parse_transport_result(result, direction)
            parsed["solver_name"] = solver_name
            parsed["runtime_s"] = time.perf_counter() - start
            return parsed
        except Exception as exc:
            errors.append(f"{solver_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("PuMA conductivity calls failed:\n" + "\n".join(errors))


def field_shape(field: Any) -> str:
    if field is None:
        return "None"
    if isinstance(field, (tuple, list)):
        return "[" + ", ".join(str(np.asarray(item).shape) for item in field) + "]"
    return str(np.asarray(field).shape)


def center_slice(array: np.ndarray, direction: str = "z") -> np.ndarray:
    if direction == "x":
        return array[array.shape[0] // 2, :, :]
    if direction == "y":
        return array[:, array.shape[1] // 2, :]
    return array[:, :, array.shape[2] // 2]


def orthogonal_center_slices(array: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(array)
    return {
        "yz_center_x": arr[arr.shape[0] // 2, :, :],
        "xz_center_y": arr[:, arr.shape[1] // 2, :],
        "xy_center_z": arr[:, :, arr.shape[2] // 2],
    }


def save_flux_slice_png(flux_mag: np.ndarray | None, path: Path, title: str, direction: str, log_scale: bool = False,
                        vmin: float | None = None, vmax: float | None = None) -> None:
    if flux_mag is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image = center_slice(np.asarray(flux_mag), direction)
    if log_scale:
        eps = max(float(np.nanmax(image)) * 1e-12, 1e-30)
        image = np.log10(image + eps)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(image.T, origin="lower", cmap="magma", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_scalar_slice_png(array: np.ndarray | None, path: Path, title: str, direction: str,
                          cmap: str = "viridis", vmin: float | None = None, vmax: float | None = None) -> None:
    if array is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image = center_slice(np.asarray(array), direction)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(image.T, origin="lower", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_scalar_orthogonal_slices(array: np.ndarray | None, out_dir: Path, prefix: str, title: str,
                                  cmap: str = "viridis") -> None:
    if array is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    for name, image in orthogonal_center_slices(arr).items():
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        im = ax.imshow(image.T, origin="lower", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title} {name}")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.savefig(out_dir / f"{prefix}_{name}.png", dpi=180)
        plt.close(fig)


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, value in arrays.items():
        if value is None:
            continue
        serializable[key] = np.asarray(value)
    np.savez_compressed(path, **serializable)


def current_memory_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / 1024**2)
    except Exception:
        pass
    if platform.system() != "Windows":
        try:
            import resource

            return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        except Exception:
            pass
    return float("nan")


def write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def phase_fraction_column(phase: str) -> str:
    return f"{phase} fraction"

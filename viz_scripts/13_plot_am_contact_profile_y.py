#!/usr/bin/env python
"""Plot y-binned AM contact-area fractions.

For each depth bin, this script counts AM-SE and AM-Void/C-rich face contacts
and normalizes them by the total internal AM contact area in that bin:

    AM-SE / (AM-SE + AM-Void/C)
    AM-Void/C / (AM-SE + AM-Void/C)

The distance axis follows the current convention used in the phase-fraction
profiles: the current collector is at the current top of the volume, so the
original y coordinate is reversed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd
import tifffile as tiff

from viz_style import apply_publication_style, panel_figsize, save_figure, style_axes, style_colorbar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ("WM", "PFDT")
SAMPLE_LABELS = {
    "WM": "WM-LPSCB",
    "PFDT": "PFDT@LPSCB",
}
SAMPLE_COLORS = {
    "WM": "#689BCA",
    "PFDT": "#EC716A",
}
SE_STACK_COLOR = "#d8d8d8"
AM_LABEL = 1
SE_LABEL = 2
VOID_LABEL = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot y-binned AM contact-area fractions.")
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument(
        "--representative-root",
        default=PROJECT_ROOT / "server_results" / "30um-in-plane" / "results",
    )
    parser.add_argument(
        "--dataset",
        choices=["whole_roi", "representative30"],
        default="whole_roi",
        help="Dataset to profile. whole_roi uses full label_zyx.tif; representative30 uses metadata crop.",
    )
    parser.add_argument("--samples", nargs="+", default=list(SAMPLES), choices=list(SAMPLES))
    parser.add_argument("--voxel-size-um", type=float, default=0.07)
    parser.add_argument("--bin-um", type=float, default=2.0)
    parser.add_argument(
        "--contact-mode",
        choices=["face1", "majority"],
        default="face1",
        help="face1 counts actual AM-SE/AM-Void shared faces; majority classifies AM surface voxels by local majority.",
    )
    parser.add_argument("--majority-radius", type=int, default=2)
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "viz" / "geometry_metrics")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


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


def load_label_zyx(args: argparse.Namespace, sample: str):
    path = label_path(Path(args.project_root), sample)
    if args.dataset == "whole_roi":
        return tiff.memmap(path)

    meta = read_json(Path(args.representative_root) / sample / "representative_flux" / "metadata.json")
    label_zyx = tiff.imread(path).astype(np.uint8, copy=False)
    return crop_label_to_metadata(label_zyx, meta)


def bin_edges_for_y(y_count: int, voxel_size_um: float, bin_um: float) -> np.ndarray:
    y_length_um = y_count * voxel_size_um
    edges = np.arange(0.0, y_length_um, bin_um)
    if edges.size == 0 or edges[0] != 0.0:
        edges = np.insert(edges, 0, 0.0)
    if edges[-1] < y_length_um:
        edges = np.append(edges, y_length_um)
    else:
        edges[-1] = y_length_um
    return edges


def bin_counts(distance_um: np.ndarray, counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bin_index = np.searchsorted(edges, distance_um, side="right") - 1
    bin_index = np.clip(bin_index, 0, len(edges) - 2)
    return np.bincount(bin_index, weights=counts.astype(np.float64), minlength=len(edges) - 1)


def add_face_contact_counts(
    out_se: np.ndarray,
    out_void: np.ndarray,
    label_zyx: np.ndarray,
    axis: int,
    edges: np.ndarray,
    voxel_size_um: float,
) -> None:
    left = [slice(None)] * 3
    right = [slice(None)] * 3
    left[axis] = slice(None, -1)
    right[axis] = slice(1, None)
    a = label_zyx[tuple(left)]
    b = label_zyx[tuple(right)]

    se_contact = ((a == AM_LABEL) & (b == SE_LABEL)) | ((a == SE_LABEL) & (b == AM_LABEL))
    void_contact = ((a == AM_LABEL) & (b == VOID_LABEL)) | ((a == VOID_LABEL) & (b == AM_LABEL))

    y_count = label_zyx.shape[1]
    y_length_um = y_count * voxel_size_um
    if axis == 1:
        old_y_um = (np.arange(y_count - 1, dtype=np.float64) + 1.0) * voxel_size_um
        count_axes = (0, 2)
    else:
        old_y_um = (np.arange(y_count, dtype=np.float64) + 0.5) * voxel_size_um
        count_axes = (0, 2)

    distance_um = y_length_um - old_y_um
    se_by_y = np.count_nonzero(se_contact, axis=count_axes)
    void_by_y = np.count_nonzero(void_contact, axis=count_axes)
    out_se += bin_counts(distance_um, se_by_y, edges)
    out_void += bin_counts(distance_um, void_by_y, edges)


def face_contact_profile(label_zyx: np.ndarray, voxel_size_um: float, bin_um: float) -> pd.DataFrame:
    edges = bin_edges_for_y(label_zyx.shape[1], voxel_size_um, bin_um)
    y_length_um = label_zyx.shape[1] * voxel_size_um
    se_counts = np.zeros(len(edges) - 1, dtype=np.float64)
    void_counts = np.zeros(len(edges) - 1, dtype=np.float64)
    for axis in range(3):
        add_face_contact_counts(se_counts, void_counts, label_zyx, axis, edges, voxel_size_um)

    face_area_um2 = voxel_size_um ** 2
    total_counts = se_counts + void_counts
    centers = 0.5 * (edges[:-1] + edges[1:])
    return pd.DataFrame(
        {
            "distance_from_current_collector_um": centers,
            "roi_y_length_um": y_length_um,
            "am_se_contact_area_um2": se_counts * face_area_um2,
            "am_void_contact_area_um2": void_counts * face_area_um2,
            "am_internal_contact_area_um2": total_counts * face_area_um2,
            "am_se_fraction_of_am_contact_area": np.divide(
                se_counts, total_counts, out=np.full_like(se_counts, np.nan), where=total_counts > 0
            ),
            "am_void_fraction_of_am_contact_area": np.divide(
                void_counts, total_counts, out=np.full_like(void_counts, np.nan), where=total_counts > 0
            ),
        }
    )


def six_neighbor(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=bool)
    out[:-1, :, :] |= mask[1:, :, :]
    out[1:, :, :] |= mask[:-1, :, :]
    out[:, :-1, :] |= mask[:, 1:, :]
    out[:, 1:, :] |= mask[:, :-1, :]
    out[:, :, :-1] |= mask[:, :, 1:]
    out[:, :, 1:] |= mask[:, :, :-1]
    return out


def local_phase_count(mask: np.ndarray, radius: int) -> np.ndarray:
    try:
        from scipy import ndimage

        kernel = np.ones((2 * radius + 1, 2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        return ndimage.convolve(mask.astype(np.uint8, copy=False), kernel, mode="constant", cval=0)
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


def majority_contact_profile(
    label_zyx: np.ndarray,
    voxel_size_um: float,
    bin_um: float,
    radius: int,
) -> pd.DataFrame:
    edges = bin_edges_for_y(label_zyx.shape[1], voxel_size_um, bin_um)
    y_length_um = label_zyx.shape[1] * voxel_size_um
    old_y_um = (np.arange(label_zyx.shape[1], dtype=np.float64) + 0.5) * voxel_size_um
    distance_um = y_length_um - old_y_um

    am = label_zyx == AM_LABEL
    surface = am & six_neighbor(~am)
    se_count = local_phase_count(label_zyx == SE_LABEL, radius)
    void_count = local_phase_count(label_zyx == VOID_LABEL, radius)
    classified = surface & ((se_count + void_count) > 0)
    void_contact = classified & (void_count > se_count)
    se_contact = classified & ~void_contact

    se_by_y = np.count_nonzero(se_contact, axis=(0, 2))
    void_by_y = np.count_nonzero(void_contact, axis=(0, 2))
    total_by_y = np.count_nonzero(surface, axis=(0, 2))
    se_counts = bin_counts(distance_um, se_by_y, edges)
    void_counts = bin_counts(distance_um, void_by_y, edges)
    surface_counts = bin_counts(distance_um, total_by_y, edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return pd.DataFrame(
        {
            "distance_from_current_collector_um": centers,
            "roi_y_length_um": y_length_um,
            "am_se_surface_voxel_count": se_counts,
            "am_void_surface_voxel_count": void_counts,
            "am_surface_voxel_count": surface_counts,
            "am_se_fraction_of_am_contact_area": np.divide(
                se_counts, surface_counts, out=np.full_like(se_counts, np.nan), where=surface_counts > 0
            ),
            "am_void_fraction_of_am_contact_area": np.divide(
                void_counts, surface_counts, out=np.full_like(void_counts, np.nan), where=surface_counts > 0
            ),
        }
    )


def load_profiles(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for sample in args.samples:
        print(f"{args.dataset} {sample}: loading labels")
        label_zyx = load_label_zyx(args, sample)
        print(f"{args.dataset} {sample}: shape zyx={tuple(label_zyx.shape)}")
        if args.contact_mode == "face1":
            profile = face_contact_profile(label_zyx, float(args.voxel_size_um), float(args.bin_um))
        else:
            profile = majority_contact_profile(
                label_zyx,
                float(args.voxel_size_um),
                float(args.bin_um),
                int(args.majority_radius),
            )
        profile.insert(0, "sample", sample)
        profile.insert(0, "dataset", args.dataset)
        profile.insert(2, "sample_label", SAMPLE_LABELS[sample])
        frames.append(profile)
    return pd.concat(frames, ignore_index=True)


def plot_profiles(df: pd.DataFrame, out_path: Path, title_suffix: str, *, dpi: int) -> None:
    apply_publication_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=panel_figsize(ncols=2, panel_width_mm=66.80, panel_height_mm=53.97, extra_width_mm=10),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    panels = (
        ("am_se_fraction_of_am_contact_area", "AM-SE / AM surface"),
        ("am_void_fraction_of_am_contact_area", "AM-Void/C / AM surface"),
    )
    for ax, (column, ylabel) in zip(axes, panels):
        for sample in SAMPLES:
            sample_df = df[df["sample"] == sample].sort_values("distance_from_current_collector_um")
            if sample_df.empty:
                continue
            x = sample_df["distance_from_current_collector_um"].to_numpy()
            y = sample_df[column].to_numpy()
            ax.plot(
                x,
                y,
                color=SAMPLE_COLORS[sample],
                linewidth=1.2,
                marker="o",
                markersize=2.4,
                markeredgewidth=0,
                label=SAMPLE_LABELS[sample],
            )
            ax.fill_between(x, y, color=SAMPLE_COLORS[sample], alpha=0.10, linewidth=0)
        ax.set_xlabel("Distance from current collector (μm)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.02)
        ax.set_title(title_suffix)
        style_axes(ax, minor=True)

    axes[0].legend(loc="best", handlelength=1.4)
    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)


def plot_void_contact_profile(df: pd.DataFrame, out_path: Path, *, dpi: int) -> None:
    apply_publication_style()
    fig, ax = plt.subplots(
        1,
        1,
        figsize=panel_figsize(ncols=1, panel_width_mm=66.80, panel_height_mm=53.97, extra_width_mm=8),
        constrained_layout=True,
    )
    for sample in SAMPLES:
        sample_df = df[df["sample"] == sample].sort_values("distance_from_current_collector_um")
        if sample_df.empty:
            continue
        x = sample_df["distance_from_current_collector_um"].to_numpy()
        y = 100.0 * sample_df["am_void_fraction_of_am_contact_area"].to_numpy()
        ax.plot(
            x,
            y,
            color=SAMPLE_COLORS[sample],
            linewidth=1.3,
            marker="o",
            markersize=2.3,
            markeredgewidth=0,
            label=SAMPLE_LABELS[sample],
        )
        ax.fill_between(x, y, color=SAMPLE_COLORS[sample], alpha=0.13, linewidth=0)

    ax.set_xlabel("Distance from current collector (μm)")
    ax.set_ylabel("AM-Void/C / AM surface")
    x_max = float(np.nanmax(df["roi_y_length_um"])) if "roi_y_length_um" in df else 45.0
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 25)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.legend(loc="upper left", handlelength=1.4)
    style_axes(ax, minor=False, x_major=False, y_major=False)
    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)


def plot_binmaps(df: pd.DataFrame, out_path: Path, title_suffix: str, *, dpi: int) -> None:
    apply_publication_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=panel_figsize(ncols=2, panel_width_mm=66.80, panel_height_mm=30.0, extra_width_mm=22),
        constrained_layout=True,
        sharex=True,
    )
    panels = (
        ("am_se_fraction_of_am_contact_area", "AM-SE / AM surface", "#376795"),
        ("am_void_fraction_of_am_contact_area", "AM-Void/C / AM surface", "#EC716A"),
    )

    samples = [sample for sample in SAMPLES if sample in set(df["sample"])]
    x = (
        df[df["sample"] == samples[0]]
        .sort_values("distance_from_current_collector_um")["distance_from_current_collector_um"]
        .to_numpy()
    )
    if len(x) > 1:
        dx = float(np.median(np.diff(x)))
    else:
        dx = 1.0
    x_edges = np.r_[x - 0.5 * dx, x[-1] + 0.5 * dx]
    y_edges = np.arange(len(samples) + 1, dtype=float)

    for ax, (column, title, color) in zip(axes, panels):
        values = []
        for sample in samples:
            sample_df = df[df["sample"] == sample].sort_values("distance_from_current_collector_um")
            values.append(sample_df[column].to_numpy())
        values_array = np.asarray(values, dtype=np.float64)
        finite = values_array[np.isfinite(values_array)]
        if finite.size:
            vmin = max(0.0, np.floor(float(finite.min()) / 0.05) * 0.05)
            vmax = min(1.0, np.ceil(float(finite.max()) / 0.05) * 0.05)
            if vmax <= vmin:
                vmax = min(1.0, vmin + 0.05)
        else:
            vmin, vmax = 0.0, 1.0
        cmap = LinearSegmentedColormap.from_list(
            f"{column}_fraction",
            ["#ffffff", color],
        )
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            values_array,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )
        ax.set_xlabel("Distance from current collector (μm)")
        ax.set_yticks(np.arange(len(samples)) + 0.5)
        ax.set_yticklabels([SAMPLE_LABELS[sample] for sample in samples])
        ax.set_title(f"{title} ({title_suffix})")
        style_axes(ax, minor=True, y_major=False)
        cbar = fig.colorbar(image, ax=ax, fraction=0.048, pad=0.025)
        cbar.set_label("Fraction")
        style_colorbar(cbar)

    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)


def plot_stacked_bar_for_sample(
    df: pd.DataFrame,
    sample: str,
    out_path: Path,
    *,
    title_suffix: str,
    dpi: int,
) -> None:
    apply_publication_style()
    sample_df = df[df["sample"] == sample].sort_values("distance_from_current_collector_um")
    x = sample_df["distance_from_current_collector_um"].to_numpy()
    se_fraction = sample_df["am_se_fraction_of_am_contact_area"].to_numpy()
    void_fraction = sample_df["am_void_fraction_of_am_contact_area"].to_numpy()
    width = float(np.median(np.diff(x))) * 0.86 if len(x) > 1 else 0.8

    fig, ax = plt.subplots(
        1,
        1,
        figsize=panel_figsize(ncols=1, panel_width_mm=66.80, panel_height_mm=53.97, extra_width_mm=8),
        constrained_layout=True,
    )
    ax.bar(
        x,
        void_fraction,
        width=width,
        color=SAMPLE_COLORS[sample],
        edgecolor="none",
        label="AM-Void/C",
        align="center",
    )
    ax.bar(
        x,
        se_fraction,
        width=width,
        bottom=void_fraction,
        color=SE_STACK_COLOR,
        edgecolor="none",
        label="AM-SE",
        align="center",
    )
    ax.set_title(f"{SAMPLE_LABELS[sample]} {title_suffix}")
    ax.set_xlabel("Distance from current collector (μm)")
    ax.set_ylabel("Fraction of AM surface")
    ax.set_xlim(max(0.0, float(x.min() - 0.5 * width)), float(x.max() + 0.5 * width))
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", handlelength=1.2)
    style_axes(ax, minor=True)
    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)


def plot_stacked_bar_combined(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title_suffix: str,
    dpi: int,
) -> None:
    apply_publication_style()
    samples = [sample for sample in SAMPLES if sample in set(df["sample"])]
    fig, axes = plt.subplots(
        len(samples),
        1,
        figsize=panel_figsize(
            ncols=1,
            nrows=len(samples),
            panel_width_mm=66.80,
            panel_height_mm=32.0,
            extra_width_mm=8,
        ),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    if len(samples) == 1:
        axes = [axes]

    for ax, sample in zip(axes, samples):
        sample_df = df[df["sample"] == sample].sort_values("distance_from_current_collector_um")
        x = sample_df["distance_from_current_collector_um"].to_numpy()
        se_fraction = sample_df["am_se_fraction_of_am_contact_area"].to_numpy()
        void_fraction = sample_df["am_void_fraction_of_am_contact_area"].to_numpy()
        width = float(np.median(np.diff(x))) * 0.86 if len(x) > 1 else 0.8
        ax.bar(
            x,
            void_fraction,
            width=width,
            color=SAMPLE_COLORS[sample],
            edgecolor="none",
            label="AM-Void/C",
            align="center",
        )
        ax.bar(
            x,
            se_fraction,
            width=width,
            bottom=void_fraction,
            color=SE_STACK_COLOR,
            edgecolor="none",
            label="AM-SE",
            align="center",
        )
        ax.text(
            0.01,
            0.92,
            SAMPLE_LABELS[sample],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )
        ax.set_ylabel("Fraction")
        ax.set_ylim(0, 1.0)
        style_axes(ax, minor=True)

    axes[-1].set_xlabel("Distance from current collector (μm)")
    axes[0].set_title(f"AM surface contact state ({title_suffix})")
    axes[0].legend(
        handles=[
            Patch(facecolor=SAMPLE_COLORS["WM"], edgecolor="none", label="WM AM-Void/C"),
            Patch(facecolor=SAMPLE_COLORS["PFDT"], edgecolor="none", label="PFDT AM-Void/C"),
            Patch(facecolor=SE_STACK_COLOR, edgecolor="none", label="AM-SE"),
        ],
        loc="upper right",
        handlelength=1.2,
    )
    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_df = load_profiles(args)

    mode_token = args.contact_mode if args.contact_mode == "face1" else f"majority_r{int(args.majority_radius)}"
    bin_token = f"bin{float(args.bin_um):g}um"
    stem = f"{args.dataset}_am_contact_fraction_profile_y_{mode_token}_{bin_token}"
    csv_path = out_dir / f"{stem}.csv"
    png_path = out_dir / f"{stem}.png"
    void_only_path = out_dir / f"{stem}_void_only.png"
    binmap_path = out_dir / f"{stem}_binmap.png"
    profile_df.to_csv(csv_path, index=False)
    plot_profiles(profile_df, png_path, title_suffix=mode_token, dpi=args.dpi)
    plot_void_contact_profile(profile_df, void_only_path, dpi=args.dpi)
    plot_binmaps(profile_df, binmap_path, title_suffix=mode_token, dpi=args.dpi)
    stacked_paths = []
    for sample in args.samples:
        sample_stacked_path = out_dir / f"{stem}_stacked_bar_{sample}.png"
        plot_stacked_bar_for_sample(
            profile_df,
            sample,
            sample_stacked_path,
            title_suffix=mode_token,
            dpi=args.dpi,
        )
        stacked_paths.append(sample_stacked_path)
    combined_stacked_path = out_dir / f"{stem}_stacked_bar_combined.png"
    plot_stacked_bar_combined(
        profile_df,
        combined_stacked_path,
        title_suffix=mode_token,
        dpi=args.dpi,
    )
    print(f"Saved {csv_path}")
    print(f"Saved {png_path}")
    print(f"Saved {void_only_path}")
    print(f"Saved {binmap_path}")
    for path in stacked_paths:
        print(f"Saved {path}")
    print(f"Saved {combined_stacked_path}")


if __name__ == "__main__":
    main()

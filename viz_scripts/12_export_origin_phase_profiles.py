#!/usr/bin/env python
"""Export selected Matplotlib-derived datasets to OriginPro projects.

This script intentionally does not import or modify the existing plotting
scripts. It rebuilds the data used by the whole-ROI phase-fraction profile,
writes CSV fallbacks, creates Origin worksheets/graph pages, and saves an
editable .opju project.

Run from the repository root after OriginPro is installed and licensed:

    python viz_scripts/12_export_origin_phase_profiles.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "viz" / "origin_projects"

SAMPLE_LABELS = {
    "WM": "WM-LPSCB",
    "PFDT": "PFDT@LPSCB",
}

PHASES = {
    "CAM": 1,
    "SE-rich": 2,
    "Void/C-rich": 3,
}

PHASE_COLORS = {
    "CAM": "#376795",
    "SE-rich": "#72bcd5",
    "Void/C-rich": "#ffd06f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an OriginPro .opju project from current plotting data."
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=["WM", "PFDT"],
        choices=sorted(SAMPLE_LABELS),
        help="Samples to export.",
    )
    parser.add_argument(
        "--voxel-size-um",
        type=float,
        default=0.07,
        help="Voxel size of the whole-ROI label volumes.",
    )
    parser.add_argument(
        "--bin-um",
        type=float,
        default=2.0,
        help="Y-bin thickness for phase-fraction profiles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for .opju and CSV outputs.",
    )
    parser.add_argument(
        "--opju-name",
        default="whole_roi_phase_fraction_profiles.opju",
        help="Origin project filename.",
    )
    parser.add_argument(
        "--hide-origin",
        action="store_true",
        help="Hide Origin while exporting. Useful for batch runs.",
    )
    parser.add_argument(
        "--keep-origin-open",
        action="store_true",
        help="Leave Origin open after saving the project.",
    )
    return parser.parse_args()


def label_path(sample: str) -> Path:
    path = RESULTS_DIR / sample / "label_zyx.tif"
    if not path.exists():
        raise FileNotFoundError(f"Missing label volume for {sample}: {path}")
    return path


def phase_profile_for_label(
    labels: np.ndarray,
    *,
    voxel_size_um: float,
    bin_um: float,
) -> pd.DataFrame:
    """Compute y-direction phase fractions with top treated as current collector.

    The label array is z-y-x. The previously plotted y coordinate was measured
    from the array origin. Here we reverse it so distance=0 is the current
    collector at the current top of the volume.
    """

    y_count = labels.shape[1]
    y_length_um = y_count * voxel_size_um
    y_centers_um = (np.arange(y_count) + 0.5) * voxel_size_um
    distance_from_cc_um = y_length_um - y_centers_um

    bins = np.arange(0.0, y_length_um + bin_um, bin_um)
    if bins[-1] < y_length_um:
        bins = np.append(bins, y_length_um)

    records: list[dict[str, float]] = []
    for start, stop in zip(bins[:-1], bins[1:]):
        in_bin = (distance_from_cc_um >= start) & (distance_from_cc_um < stop)
        if not np.any(in_bin):
            continue

        slab = labels[:, in_bin, :]
        total = float(slab.size)
        row = {
            "Distance from current collector": 0.5 * (start + stop),
        }
        for phase_name, phase_value in PHASES.items():
            row[phase_name] = 100.0 * float(np.count_nonzero(slab == phase_value)) / total
        records.append(row)

    return pd.DataFrame.from_records(records)


def load_profiles(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    profiles: dict[str, pd.DataFrame] = {}
    for sample in args.samples:
        labels = tiff.imread(label_path(sample))
        profiles[sample] = phase_profile_for_label(
            labels,
            voxel_size_um=args.voxel_size_um,
            bin_um=args.bin_um,
        )
    return profiles


def write_csv_fallbacks(output_dir: Path, profiles: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample, df in profiles.items():
        out = output_dir / f"whole_roi_phase_fraction_profile_from_current_collector_{sample}.csv"
        df.to_csv(out, index=False)


def set_origin_visible(op, visible: bool) -> None:
    try:
        op.set_show(visible)
    except Exception:
        pass


def new_origin_graph(op, lname: str):
    """Create a graph page, preferring stacked-column templates if available."""

    for template in ("origin", "StackCol", "ColumnStack", "Column", ""):
        try:
            graph = op.new_graph(lname=lname, template=template)
            if graph is not None:
                return graph
        except Exception:
            continue
    return None


def configure_origin_axis(layer, x_max: float) -> None:
    try:
        layer.axis("x").title = "Distance from current collector (um)"
        layer.axis("y").title = "Volume fraction (%)"
    except Exception:
        pass
    try:
        layer.xlim = (0, float(np.ceil(x_max / 10.0) * 10.0))
        layer.ylim = (0, 100)
    except Exception:
        pass
    try:
        layer.rescale()
    except Exception:
        pass


def add_phase_profile_graph(op, worksheet, sample: str, df: pd.DataFrame) -> None:
    graph = new_origin_graph(op, f"{sample}_phase_fraction_profile")
    if graph is None:
        print(f"Warning: Origin graph page could not be created for {sample}; worksheet was still exported.")
        return
    layer = graph[0]

    plots = []
    for y_col, phase_name in enumerate(PHASES, start=1):
        plot = layer.add_plot(worksheet, coly=y_col, colx=0, type="c")
        if plot is not None:
            try:
                plot.color = PHASE_COLORS[phase_name]
            except Exception:
                pass
            plots.append(plot)

    if len(plots) > 1:
        try:
            layer.group(True, 0, len(plots) - 1)
        except Exception:
            pass

    configure_origin_axis(layer, df["Distance from current collector"].max())

    # Origin template availability varies by installation. The worksheets remain
    # editable even if the local template falls back to grouped columns.
    try:
        layer.obj.LT_execute("layer -g;")
    except Exception:
        pass


def export_origin_project(
    output_dir: Path,
    opju_name: str,
    profiles: dict[str, pd.DataFrame],
    *,
    show_origin: bool,
    keep_origin_open: bool,
) -> Path:
    try:
        import originpro as op
    except ImportError as exc:
        raise RuntimeError(
            "originpro is not importable. Install OriginPro's Python package or run "
            "this script from the Python environment connected to Origin."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    opju_path = output_dir / opju_name

    set_origin_visible(op, show_origin)
    op.new(asksave=False)

    workbook = op.new_book(lname="whole_roi_phase_fraction_profiles")
    for index, (sample, df) in enumerate(profiles.items()):
        if index == 0:
            worksheet = workbook[0]
        else:
            worksheet = workbook.add_sheet(sample)

        try:
            worksheet.name = sample
        except Exception:
            pass

        worksheet.from_list(
            0,
            df["Distance from current collector"].to_numpy(),
            lname="Distance from current collector",
            units="um",
            axis="X",
        )
        for col_index, phase_name in enumerate(PHASES, start=1):
            worksheet.from_list(
                col_index,
                df[phase_name].to_numpy(),
                lname=phase_name,
                units="%",
                axis="Y",
            )

        add_phase_profile_graph(op, worksheet, sample, df)

    op.save(str(opju_path))
    if not keep_origin_open:
        try:
            op.exit()
        except Exception:
            pass
    return opju_path


def main() -> None:
    args = parse_args()
    profiles = load_profiles(args)
    write_csv_fallbacks(args.output_dir, profiles)
    opju_path = export_origin_project(
        args.output_dir,
        args.opju_name,
        profiles,
        show_origin=not args.hide_origin,
        keep_origin_open=args.keep_origin_open,
    )
    print(f"Saved Origin project: {opju_path}")
    print(f"Saved CSV fallbacks in: {args.output_dir}")


if __name__ == "__main__":
    main()

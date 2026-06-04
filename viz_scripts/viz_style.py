from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


MM_PER_IN = 25.4
PANEL_HEIGHT_MM = 53.97
PANEL_WIDTH_MM = 85.0
NICE_TICK_STEPS = [1, 2, 5, 10]


def mm_to_in(value_mm: float) -> float:
    return value_mm / MM_PER_IN


def panel_figsize(
    ncols: int = 1,
    nrows: int = 1,
    panel_width_mm: float = PANEL_WIDTH_MM,
    panel_height_mm: float = PANEL_HEIGHT_MM,
    extra_width_mm: float = 0.0,
    extra_height_mm: float = 0.0,
) -> tuple[float, float]:
    margin_width_mm = 22.0 + 12.0 * max(0, ncols - 1)
    margin_height_mm = 18.0 + 12.0 * max(0, nrows - 1)
    return (
        mm_to_in(panel_width_mm * ncols + margin_width_mm + extra_width_mm),
        mm_to_in(panel_height_mm * nrows + margin_height_mm + extra_height_mm),
    )


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.linewidth": 0.9,
            "axes.grid": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.size": 1.6,
            "ytick.minor.size": 1.6,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.width": 0.7,
            "ytick.minor.width": 0.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.edgecolor": "none",
            "savefig.transparent": True,
        }
    )


def _iter_axes(axes) -> Iterable:
    if hasattr(axes, "flat"):
        yield from axes.flat
    elif isinstance(axes, (list, tuple)):
        for item in axes:
            yield from _iter_axes(item)
    else:
        yield axes


def style_axes(axes, *, minor: bool = True, grid: bool = False, x_major: bool = True, y_major: bool = True) -> None:
    for ax in _iter_axes(axes):
        ax.grid(grid)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
        if x_major:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=NICE_TICK_STEPS))
        if y_major:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=NICE_TICK_STEPS))
        ax.tick_params(axis="both", which="major", labelsize=9, length=3, width=0.9, direction="out")
        ax.tick_params(axis="both", which="minor", length=1.6, width=0.7, direction="out")
        if minor:
            try:
                if x_major:
                    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                if y_major:
                    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            except Exception:
                ax.minorticks_on()


def style_colorbar(cbar) -> None:
    cbar.locator = MaxNLocator(nbins=4, steps=NICE_TICK_STEPS)
    cbar.update_ticks()
    cbar.ax.tick_params(which="major", labelsize=9, length=3, width=0.9, direction="out")
    cbar.outline.set_linewidth(0.9)
    cbar.ax.yaxis.label.set_size(10)
    cbar.ax.xaxis.label.set_size(10)


def save_figure(fig, path: Path, *, dpi: int = 600) -> None:
    fig.savefig(
        path,
        dpi=dpi,
        transparent=True,
        facecolor="none",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.02,
    )

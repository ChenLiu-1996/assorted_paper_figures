"""High-level helper utilities for publication-style scientific figures.

This module codifies the dominant design theory extracted from the figures4papers repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

GROUPED_SERIES_COLORS = [
    PALETTE["blue_main"],
    PALETTE["green_3"],
    PALETTE["red_strong"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["neutral"],
]


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    use_tex: bool = False


def apply_publication_style(style: FigureStyle = FigureStyle()) -> None:
    """Apply repository-aligned matplotlib rcParams."""
    plt.rcParams.update(
        {
            "text.usetex": style.use_tex,
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": style.font_size,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": style.axes_linewidth,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def finalize_figure(fig: plt.Figure, out_path: str, dpi: int = 300, pad: float = 2.0) -> None:
    """Finalize and save using repository defaults, ensuring no cropping."""
    save_path = Path(out_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use tight_layout with specified padding
    fig.tight_layout(pad=pad)
    
    # Save with bbox_inches='tight' and pad_inches=0.1 to prevent cropping
    fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight', pad_inches=0.1)


def make_sphere_illustration(ax: plt.Axes, light_dir: np.ndarray | None = None) -> None:
    """
    Draw a shaded 3D-effect sphere using a meshgrid and intensity map.
    Discovery from repo: This uses a 2D imshow with a calculated mask and shading
    to simulate a 3D sphere with publication-grade lighting.
    """
    res = 512
    xs = np.linspace(-1, 1, res)
    ys = np.linspace(-1, 1, res)
    x, y = np.meshgrid(xs, ys)
    r2 = x**2 + y**2
    mask = r2 <= 1.0
    
    z = np.zeros_like(x)
    z[mask] = np.sqrt(1.0 - r2[mask])
    
    # Normal vectors
    norm = np.sqrt(x**2 + y**2 + z**2) + 1e-6
    nx, ny, nz = x/norm, y/norm, z/norm
    
    if light_dir is None:
        light_dir = np.array([-0.5, 0.5, 0.8])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    intensity = np.maximum(0.0, nx*light_dir[0] + ny*light_dir[1] + nz*light_dir[2])
    ambient = 0.3
    shade = np.clip(ambient + 0.9*intensity, 0, 1)
    
    img = np.ones((res, res))
    img[mask] = shade[mask]
    
    ax.imshow(img, cmap='gray', origin='lower', extent=[-1, 1, -1, 1], vmin=0, vmax=1, alpha=0.5)
    ax.set_axis_off()


def annotate_bars(ax: plt.Axes, bars: plt.BarContainer, fontsize: int = 12) -> None:
    """Add numeric labels above bars for precision reading."""
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.2f}', ha='center', va='bottom', fontsize=fontsize)


def make_grouped_bar(
    ax: plt.Axes,
    categories: Sequence[str],
    series: Sequence[Sequence[float]],
    labels: Sequence[str],
    colors: Sequence[str] | None = None,
    y_label: str = "Score",
) -> None:
    """Draw grouped bars with black edges and publication-safe styling."""
    if colors is None:
        colors = GROUPED_SERIES_COLORS

    values = np.asarray(series, dtype=float)
    n_series, n_cats = values.shape
    x = np.arange(n_cats)
    width = 0.8 / n_series

    for i in range(n_series):
        ax.bar(
            x + (i - (n_series - 1) / 2) * width,
            values[i],
            width=width,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=1.5,
            label=labels[i],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=0)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, max(1.0, float(values.max()) * 1.12))
    ax.legend(loc="upper left")


def make_trend(
    ax: plt.Axes,
    x: Sequence[float],
    y_series: Sequence[Sequence[float]],
    labels: Sequence[str],
    colors: Sequence[str] | None = None,
    y_label: str = "Value",
    x_label: str = "Step",
    show_shadow: bool = True,
) -> None:
    """Draw repository-style trend lines with soft shadows/error bands."""
    if colors is None:
        colors = [PALETTE["blue_main"], PALETTE["red_strong"], PALETTE["green_3"], PALETTE["teal"]]

    x_arr = np.asarray(x, dtype=float)
    for i, y in enumerate(y_series):
        y_arr = np.asarray(y)
        color = colors[i % len(colors)]
        
        if show_shadow:
            for offset, alpha in zip([0.02, 0.04, 0.06], [0.15, 0.1, 0.05]):
                ax.fill_between(x_arr, y_arr - offset, y_arr + offset, color=color, alpha=alpha, lw=0)
        
        ax.plot(x_arr, y_arr, linewidth=3, alpha=0.9, color=color, label=labels[i])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="best", frameon=False)


def make_scatter(
    ax: plt.Axes,
    x: Sequence[float],
    y: Sequence[float],
    groups: Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
    size: float = 80,
) -> None:
    """Draw publication-style scatter with alpha and edge contrast."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if groups is None:
        ax.scatter(x_arr, y_arr, s=size, color=PALETTE["blue_secondary"], alpha=0.75, edgecolors="black", linewidths=0.8)
    else:
        group_arr = np.asarray(groups)
        uniq = sorted(set(int(v) for v in group_arr))
        colors = [PALETTE["green_3"], PALETTE["red_2"], PALETTE["blue_main"], PALETTE["teal"]]
        for i, gid in enumerate(uniq):
            mask = group_arr == gid
            label = str(gid) if labels is None else labels[i]
            ax.scatter(x_arr[mask], y_arr[mask], s=size, color=colors[i % len(colors)], alpha=0.75, edgecolors="black", linewidths=0.8, label=label)
        ax.legend(loc="best")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=16, axes_linewidth=2.5))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Bar demo
    make_grouped_bar(axes[0], ["A", "B"], [[0.8, 0.9], [0.7, 0.85]], ["Proposed", "Baseline"])
    
    # Sphere demo
    make_sphere_illustration(axes[1])
    
    # Trend demo
    x = np.linspace(0, 10, 20)
    make_trend(axes[2], x, [np.sin(x), np.cos(x)], ["Sin", "Cos"], show_shadow=True)
    
    finalize_figure(fig, "skill_test_final.png")
    print("Skill updated and verified with skill_test_final.png")

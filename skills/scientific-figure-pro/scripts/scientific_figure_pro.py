"""Publication-quality plotting utilities for scientific figures.

This module provides a suite of high-level wrappers around matplotlib to produce 
consistent, submission-ready scientific figures. It enforces a unified aesthetic 
(Helvetica-like fonts, high-contrast palettes, minimal spines) and handles 
robust export workflows for LaTeX and digital formats.

Standard usage involves:
1. Setting the global style with `apply_publication_style`.
2. Creating a figure/axes layout with `create_subplots`.
3. Populating axes using specialized `make_*` helpers.
4. Exporting via `finalize_figure`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage
    import numpy.typing as npt

# Configure logger
logger = logging.getLogger(__name__)

# --- Aesthetics & Constants ---

PALETTE: Final[dict[str, str]] = {
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

DEFAULT_COLORS: Final[list[str]] = [
    PALETTE["blue_main"],
    PALETTE["green_3"],
    PALETTE["red_strong"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["neutral"],
]

_VECTOR_FORMATS: Final[set[str]] = {"pdf", "svg", "eps"}
_RASTER_FORMATS: Final[set[str]] = {"png", "jpg", "jpeg", "tif", "tiff"}
_SUPPORTED_FORMATS: Final[set[str]] = _VECTOR_FORMATS | _RASTER_FORMATS


@dataclass(frozen=True)
class FigureStyle:
    """Configuration for global matplotlib rcParams.

    Attributes:
        font_size: Base font size in points.
        axes_linewidth: Width of the axis spines.
        use_tex: Whether to use LaTeX for text rendering.
        font_family: Priority list of font families.
    """

    font_size: int = 16
    axes_linewidth: float = 2.5
    use_tex: bool = False
    font_family: tuple[str, ...] = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")


# --- Internal Helpers ---

def _require(condition: bool, message: str) -> None:
    """Internal assertion-style validator."""
    if not condition:
        raise ValueError(f"[scientific_figure_pro] {message}")


def _as_1d_array(name: str, values: Any) -> npt.NDArray[np.float64]:
    """Ensures input is a 1D numpy array of floats."""
    arr = np.asarray(values, dtype=np.float64)
    _require(arr.ndim == 1, f"'{name}' must be 1D, got shape {arr.shape}")
    _require(arr.size > 0, f"'{name}' cannot be empty")
    return arr


def _as_2d_array(name: str, values: Any) -> npt.NDArray[np.float64]:
    """Ensures input is a 2D numpy array of floats."""
    arr = np.asarray(values, dtype=np.float64)
    _require(arr.ndim == 2, f"'{name}' must be 2D, got shape {arr.shape}")
    _require(arr.size > 0, f"'{name}' cannot be empty")
    return arr


# --- Core API ---

def apply_publication_style(style: FigureStyle | None = None) -> None:
    """Configures matplotlib with publication-ready defaults.

    Args:
        style: Optional custom FigureStyle object. Defaults to standard.
    """
    s = style or FigureStyle()

    plt.rcParams.update({
        "text.usetex": s.use_tex,
        "font.family": "sans-serif",
        "font.sans-serif": list(s.font_family),
        "font.size": s.font_size,
        "axes.labelsize": s.font_size,
        "axes.titlesize": s.font_size + 2,
        "axes.linewidth": s.axes_linewidth,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "legend.fontsize": s.font_size - 2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
    })
    logger.debug("Applied publication style configuration.")


def create_subplots(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any
) -> tuple[Figure, npt.NDArray[np.object_]]:
    """Creates a figure and a flattened array of axes.

    Args:
        nrows: Number of rows in the grid.
        ncols: Number of columns in the grid.
        figsize: (width, height) in inches.
        **kwargs: Additional arguments passed to `plt.subplots`.

    Returns:
        A tuple of (Figure, flat array of Axes).
    """
    _require(nrows > 0 and ncols > 0, "Grid dimensions must be positive integers.")
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, np.atleast_1d(np.array(axes, dtype=object)).flatten()


def finalize_figure(
    fig: Figure,
    out_path: str | Path,
    formats: Sequence[str] | None = None,
    dpi: int = 300,
    close: bool = True,
    pad: float = 0.05,
    **kwargs: Any
) -> list[Path]:
    """Saves the figure in multiple formats and closes it.

    If no formats are provided, it defaults to (pdf, svg, eps) unless the out_path
    already contains an extension.

    Args:
        fig: The matplotlib Figure object.
        out_path: Filename or directory path.
        formats: List of extensions (e.g., ['png', 'pdf']).
        dpi: Resolution for raster formats.
        close: Whether to call plt.close(fig) after saving.
        pad: Padding in inches.
        **kwargs: Passed to fig.savefig.

    Returns:
        List of Paths to the saved files.
    """
    path = Path(out_path)
    exts = formats
    
    if not exts:
        exts = [path.suffix.lstrip(".")] if path.suffix else ["pdf", "svg", "eps"]
    
    base = path.with_suffix("") if path.suffix else path
    base.parent.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for ext in exts:
        ext = ext.lower().strip(".")
        _require(ext in _SUPPORTED_FORMATS, f"Unsupported format: {ext}")
        
        target = base.with_suffix(f".{ext}")
        save_params = {"format": ext, "bbox_inches": "tight", "pad_inches": pad}
        if ext in _RASTER_FORMATS:
            save_params["dpi"] = dpi
        save_params.update(kwargs)
        
        fig.savefig(target, **save_params)
        saved.append(target)
    
    if close:
        plt.close(fig)
    
    logger.info(f"Saved figure to: {', '.join(p.name for p in saved)}")
    return saved


# --- Specialized Plotting Helpers ---

def make_trend(
    ax: Axes,
    x: Sequence[float],
    y_series: Sequence[Sequence[float]],
    labels: Sequence[str],
    colors: Sequence[str] | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    show_shadow: bool = True
) -> None:
    """Renders multiple line trends with optional confidence shadows."""
    x_arr = _as_1d_array("x", x)
    color_map = colors or DEFAULT_COLORS
    
    for i, y in enumerate(y_series):
        y_arr = _as_1d_array(f"y_series[{i}]", y)
        _require(len(x_arr) == len(y_arr), f"Length mismatch in series {i}")
        
        color = color_map[i % len(color_map)]
        if show_shadow:
            span = np.ptp(y_arr) or 1.0
            ax.fill_between(x_arr, y_arr - 0.03*span, y_arr + 0.03*span, 
                           color=color, alpha=0.1, lw=0)
        
        ax.plot(x_arr, y_arr, label=labels[i], color=color, lw=2.5, alpha=0.9)

    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.legend()


def make_grouped_bar(
    ax: Axes,
    categories: Sequence[str],
    series: Sequence[Sequence[float]],
    labels: Sequence[str],
    ylabel: str = "Value",
    colors: Sequence[str] | None = None,
    annotate: bool = False
) -> BarContainer:
    """Renders a high-contrast grouped bar chart."""
    data = _as_2d_array("series", series)
    n_series, n_cats = data.shape
    _require(len(categories) == n_cats, "Category count mismatch")
    
    x = np.arange(n_cats)
    total_width = 0.8
    width = total_width / n_series
    color_map = colors or DEFAULT_COLORS

    last_bars = None
    for i in range(n_series):
        offset = (i - (n_series - 1) / 2) * width
        bars = ax.bar(x + offset, data[i], width, label=labels[i], 
                      color=color_map[i % len(color_map)], edgecolor="white", lw=0.5)
        last_bars = bars
        if annotate:
            annotate_bars(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.legend()
    return last_bars


def annotate_bars(
    ax: Axes,
    bars: BarContainer,
    fmt: str = "{:.2f}",
    fontsize: int = 10,
    padding: float = 3
) -> None:
    """Adds text labels above bars in a BarContainer."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(fmt.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, padding),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=fontsize)


def make_heatmap(
    ax: Axes,
    matrix: Sequence[Sequence[float]],
    x_labels: Sequence[str] | None = None,
    y_labels: Sequence[str] | None = None,
    cmap: str = "magma",
    cbar_label: str | None = None,
    annotate: bool = False
) -> AxesImage:
    """Renders a cleaned heatmap with optional text annotations."""
    data = _as_2d_array("matrix", matrix)
    im = ax.imshow(data, cmap=cmap, aspect="auto", interpolation="nearest")
    
    if x_labels:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
    if y_labels:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)

    if cbar_label:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

    if annotate:
        threshold = (data.max() + data.min()) / 2
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                color = "white" if val < threshold else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)
    
    return im


def make_scatter(
    ax: Axes,
    x: Sequence[float],
    y: Sequence[float],
    label: str | None = None,
    color: str | None = None,
    size: float = 50,
    alpha: float = 0.7
) -> None:
    """Renders a publication-style scatter plot."""
    x_arr = _as_1d_array("x", x)
    y_arr = _as_1d_array("y", y)
    _require(len(x_arr) == len(y_arr), "Length mismatch in scatter")
    
    ax.scatter(x_arr, y_arr, s=size, label=label, 
               color=color or PALETTE["blue_main"], alpha=alpha, edgecolors="white", lw=0.5)
    if label:
        ax.legend()


def make_sphere_illustration(
    ax: Axes,
    light_dir: tuple[float, float, float] = (-0.5, 0.5, 0.8),
    resolution: int = 128,
    alpha: float = 0.6
) -> None:
    """Draws a shaded 2D sphere to mimic 3D lighting."""
    xs = np.linspace(-1, 1, resolution)
    ys = np.linspace(-1, 1, resolution)
    x, y = np.meshgrid(xs, ys)
    r2 = x**2 + y**2
    mask = r2 <= 1.0

    z = np.zeros_like(x)
    z[mask] = np.sqrt(1.0 - r2[mask])

    norm = np.sqrt(x**2 + y**2 + z**2) + 1e-12
    nx, ny, nz = x / norm, y / norm, z / norm

    light = np.array(light_dir)
    light = light / np.linalg.norm(light)

    intensity = np.maximum(0.0, nx * light[0] + ny * light[1] + nz * light[2])
    shade = np.clip(0.3 + 0.9 * intensity, 0.0, 1.0)

    img = np.ones((resolution, resolution), dtype=float)
    img[mask] = shade[mask]

    ax.imshow(img, cmap="gray", origin="lower", extent=[-1, 1, -1, 1], 
              vmin=0, vmax=1, alpha=alpha)
    ax.set_axis_off()


def demo() -> None:
    """Runs a verification demo for all major plot types."""
    logging.basicConfig(level=logging.INFO)
    apply_publication_style()
    
    fig, axes = create_subplots(2, 3, figsize=(18, 10))
    
    # Row 1
    x = np.linspace(0, 10, 50)
    make_trend(axes[0], x, [np.sin(x), np.cos(x)], ["Sin", "Cos"], xlabel="Time")
    make_grouped_bar(axes[1], ["A", "B", "C"], [[10, 20, 15], [12, 18, 14]], ["S1", "S2"], annotate=True)
    make_heatmap(axes[2], np.random.rand(5, 5), annotate=True, cbar_label="Intensity")
    
    # Row 2
    make_scatter(axes[3], np.random.randn(50), np.random.randn(50), label="Samples", color=PALETTE["teal"])
    make_sphere_illustration(axes[4])
    axes[5].text(0.5, 0.5, "Scientific Figure Pro\nVerification Suite", ha="center", va="center", fontsize=14)
    axes[5].set_axis_off()
    
    finalize_figure(fig, "scientific_figure_demo_full", formats=["png", "pdf"])


if __name__ == "__main__":
    demo()

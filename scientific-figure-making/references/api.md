# API Reference

This document specifies the conventions, constants, and function signatures that implementations should follow for publication-style figures. Implement these in your own code or adapt from the [figure_* demos](demos.md).

---

## Constants

### PALETTE

Semantic color map for publication figures.

```python
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}
```

Use blues for key/proposed methods, greens for improvements, reds for baselines/contrasts, neutral for background.

### DEFAULT_COLORS

Ordered list of colors used when `colors` is not passed to plot helpers: `[blue_main, green_3, red_strong, teal, violet, neutral]`.

---

## FigureStyle

```python
@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    use_tex: bool = False
    font_family: tuple[str, ...] = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")
```

- **font_size:** Base font size (use 24 for large bar panels, 15–16 for compact plots).
- **axes_linewidth:** Spine width (3 for big bars, 2 for compact).
- **use_tex:** Set `True` only when LaTeX is installed and math labels are required.
- **font_family:** Fallback list for sans-serif fonts.

---

## Style and Layout

### apply_publication_style(style=None)

Configures matplotlib rcParams: spines (top/right off), frameless legend, font family/size, vector export options. Call once before creating figures.

```python
apply_publication_style(FigureStyle(font_size=24, axes_linewidth=3))
```

### create_subplots(nrows=1, ncols=1, figsize=None, **kwargs)

Returns `(fig, axes)` with `axes` a flattened 1D array of Axes. Pass `figsize=(width, height)` in inches.

```python
fig, axes = create_subplots(2, 2, figsize=(12, 8))
```

### finalize_figure(fig, out_path, formats=None, dpi=300, close=True, pad=0.05, **kwargs)

Saves the figure to one or more formats (e.g. `['png', 'pdf']`). If `formats` is None, uses extension of `out_path` or defaults to pdf/svg/eps. Creates parent directories. Returns list of saved `Path`s.

```python
finalize_figure(fig, "output/result", formats=["png", "pdf"], dpi=300, pad=0.06)
```

---

## Plot Helpers

### make_grouped_bar(ax, categories, series, labels, ylabel='Value', colors=None, annotate=False)

Renders grouped bars. `categories`: list of x labels; `series`: list of arrays (one per group), same length as categories; `labels`: legend labels. Returns the last `BarContainer` (for use with `annotate_bars`).

```python
make_grouped_bar(ax, ["A", "B", "C"], [[1,2,3], [2,1,2]], ["X", "Y"], ylabel="Score", annotate=True)
```

### annotate_bars(ax, bars, fmt='{:.2f}', fontsize=10, padding=3)

Adds text above each bar in a `BarContainer`.

```python
bars = make_grouped_bar(ax, cats, series, labels, annotate=False)
annotate_bars(ax, bars, fmt="{:.1f}", fontsize=12)
```

### make_trend(ax, x, y_series, labels, colors=None, ylabel=None, xlabel=None, show_shadow=True)

Plots multiple lines with optional uncertainty band. `y_series`: list of 1D arrays (same length as `x`).

```python
make_trend(ax, epochs, [acc_a, acc_b], ["Model A", "Model B"], ylabel="Accuracy", show_shadow=True)
```

### make_heatmap(ax, matrix, x_labels=None, y_labels=None, cmap='magma', cbar_label=None, annotate=False)

Renders a 2D heatmap. Optionally adds colorbar label and cell annotations.

```python
make_heatmap(ax, corr_matrix, x_labels=names, y_labels=names, cbar_label="Correlation")
```

### make_scatter(ax, x, y, label=None, color=None, size=50, alpha=0.7)

Single-series scatter. `x`, `y` must be same-length 1D arrays.

```python
make_scatter(ax, x_vals, y_vals, label="Samples", color=PALETTE["blue_main"])
```

### make_sphere_illustration(ax, light_dir=(-0.5, 0.5, 0.8), resolution=128, alpha=0.6)

Draws a shaded 2D disk to mimic a 3D sphere. Use for conceptual diagrams.

```python
make_sphere_illustration(ax, light_dir=(-0.55, 0.65, 0.55), resolution=280)
```

---

## Validation Rules

- All numeric sequences are converted to numpy arrays; 1D/2D and length checks are enforced.
- `make_grouped_bar`: `len(categories)` must equal number of columns in `series`.
- `make_trend`: each series must have same length as `x`.
- `finalize_figure`: supported formats are pdf, svg, eps, png, jpg, jpeg, tif, tiff.

See [design-theory.md](design-theory.md) for visual and export conventions.

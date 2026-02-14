---
name: scientific-figure-pro
description: Generate publication-ready scientific figures in Python/matplotlib with a consistent house style (Helvetica/Arial-like sans fonts, high-contrast palettes, minimal spines, frameless legends, and tight export defaults). Use when creating bar, trend, scatter, or multi-panel academic figures that must match repository-style aesthetics and reproducibility constraints.
---

# Scientific Figure Pro

Use the helpers in `scripts/scientific_figure_pro.py` to keep figure style consistent across tasks.

## Apply Style

1. Import and call `apply_publication_style(...)` first.
2. Choose scale:
- Use `FigureStyle(font_size=24, axes_linewidth=3)` for dense bar/comparison panels.
- Use `FigureStyle(font_size=15 or 16, axes_linewidth=2)` for compact trend/scatter plots.
3. Keep `text.usetex=False` unless math typography is required and TeX is installed.

## Use High-Level Plot Functions

- `make_grouped_bar(ax, categories, series, labels, ...)`
- `annotate_bars(ax, bars, fontsize=12)`: Add exact values above bars.
- `make_trend(ax, x, y_series, labels, ...)`
- `make_scatter(ax, x, y, groups=..., labels=...)`
- `make_sphere_illustration(ax, light_dir=...)`: Create a shaded 3D-effect sphere.
- `finalize_figure(fig, out_path, dpi=300, pad=2)`

These encode the house defaults:

- **Ultra-Wide Scaling:** Use wide `figsize` (e.g., width 3-4x height) for multi-metric panels.
- **Legend Isolation:** Consider dedicating the last axis in a grid to a shared legend (`ax.set_axis_off()`).
- **Precision Ticks:** Use `FixedLocator` for clean, whole-number y-ticks.
- **Top/right spines removed**
- **Frameless legends**
- **Explicit y-limits with modest headroom**
- **Black bar edges for print-safe contrast**
- **Repository-aligned color palette**

## Reproducible Export Rules

1. Default to `dpi=300`.
2. Use `dpi=600` for dense bar-heavy panels with small text.
3. Always call `finalize_figure(...)` to enforce `tight_layout(pad=2)` before saving.

## Palette Policy

Use semantic colors from `PALETTE`:

- Blues (`blue_main`, `blue_secondary`) for key/proposed methods.
- Greens (`green_1..green_3`) for positive/improvement variants.
- Reds (`red_1`, `red_2`, `red_strong`) for contrasting baselines/ablations.
- `neutral` for references/background categories.
- `highlight` only for targeted callouts.

## Quick Start

```python
import importlib.util
from pathlib import Path
import sys

module_path = Path("skills/scientific-figure-pro/scripts/scientific_figure_pro.py")
spec = importlib.util.spec_from_file_location("scientific_figure_pro", module_path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

mod.apply_publication_style(mod.FigureStyle(font_size=16, axes_linewidth=2.5))
fig, ax = plt.subplots(figsize=(9, 5))
mod.make_grouped_bar(ax, categories, series, labels)
mod.finalize_figure(fig, "figure.png", dpi=300, pad=2)
```

If importing by package path is inconvenient, load the module by relative file path and reuse the same APIs.

---
name: scientific-figure-making
description: Generates publication-ready scientific figures in Python/matplotlib with consistent house style (Helvetica/Arial-like fonts, high-contrast palettes, minimal spines, frameless legends). Use when creating bar, trend, scatter, heatmap, or multi-panel figures for papers, slides, or reports that must match repository aesthetics and print/vector export.
version: 1.0.0
author: Chen Liu <chen.liu.cl2482@yale.edu>
license: MIT
tags: [Matplotlib, Scientific Figures, Publication Quality, Data Visualization, Bar Charts, Heatmaps, LaTeX, Research, Reproducibility]
dependencies: [matplotlib>=3.7.0, numpy>=1.24.0]
---

# Scientific Figure Making

Expert guidance for publication-ready scientific figures in Python/matplotlib: consistent house style, high-level plot helpers, and reproducible export. Implementation details, API, and examples live in the reference docs; this file is the quick reference.

**Demos and guidance** for this skill come from real figure-making scripts in the [figures4papers](https://github.com/ChenLiu-1996/figures4papers) repository (`figure_*` folders). See [references/demos.md](references/demos.md) for links to each project; the design theory and patterns were derived from those scripts.

## When to use this skill

- Creating **publication-ready** bar charts, trend plots, heatmaps, or multi-panel figures for papers or reports.
- Enforcing a **consistent house style** (fonts, colors, spines, export) across many figures.
- Needing **print- and vector-safe** output (PDF/SVG/PNG at 300 DPI or higher) with minimal manual tweaking.
- Reproducibility: same style and helpers produce the same look everywhere.

## When NOT to use / Alternatives

- **Interactive dashboards or web apps** → Use Plotly, Altair, or Bokeh.
- **Quick exploratory EDA** → Use seaborn or pandas plotting; switch to this skill when finalizing for publication.
- **Complex 3D or geographic viz** → Use Mayavi, Plotly 3D, or dedicated GIS tools.
- **Heavy design/infographics** → Use Illustrator, Inkscape, or Figma; this skill targets data-driven academic figures.

## Core concepts

- **Apply publication style first.** Set global typography and axes (font size, line width, spines off top/right, frameless legends). Choose scale by figure type: larger for dense bar/comparison panels, smaller for compact trend/scatter. Keep TeX off unless you need math and have a working TeX setup.
- **Use the high-level plot helpers** for grouped bars, trend lines with optional uncertainty, heatmaps, scatter, and a shaded sphere for conceptual diagrams. They encode house defaults (palette, scaling, legend placement). Finish with the shared export helper so layout and formats are consistent.
- **Export rules:** Default 300 DPI; use higher DPI for dense bar-heavy panels. Export uses tight bounding box and optional padding; support PNG, PDF, SVG as needed.

See [references/api.md](references/api.md) for the full API and conventions to implement.

## Workflow 1: Bar chart for method comparison

- [ ] Apply publication style with the chosen figure style (font size, line width).
- [ ] Create figure and axes; use wide aspect (e.g. width 3–4× height) if many metrics.
- [ ] Use the grouped-bar helper with categories, series, labels; turn on value annotations if desired.
- [ ] Set y-limits so differences are visible.
- [ ] Export with the finalize helper (e.g. PNG and PDF, 300 DPI).

## Workflow 2: Multi-panel figure with shared legend

- [ ] Apply publication style; create subplot grid (e.g. 1×3 or 2×2).
- [ ] Plot data on the first N-1 axes; collect legend handles/labels from one axis.
- [ ] On the last axis, turn the axis off and add a single legend with those handles/labels.
- [ ] Set titles and y-limits on data panels.
- [ ] Export with the finalize helper.

See [references/common-patterns.md](references/common-patterns.md) for layout patterns (ultra-wide panels, dedicated legend axis).

## Palette policy

Use semantic colors from the skill palette:

- Blues for key/proposed methods.
- Greens for positive/improvement variants.
- Reds for contrasting baselines/ablations.
- Neutral for references/background categories.
- Highlight only for targeted callouts.

Exact keys and hex values are in [references/api.md](references/api.md).

## Common issues

| Issue | Solution |
|-------|----------|
| LaTeX not found / usetex errors | Use style with TeX disabled (default). |
| Fonts look wrong on another machine | Rely on the default font fallback chain (DejaVu Sans, Helvetica, Arial, sans-serif). |
| Legend overlaps data | Use a dedicated legend panel: add an extra subplot, turn axis off, and put the legend there. See [references/common-patterns.md](references/common-patterns.md). |
| Bars too small or crowded | Use wider figsize (e.g. 3–4× width vs height) and/or fewer categories per row. |
| Shape/length mismatch errors | Ensure category count matches series columns; each trend series must match x length. See [references/api.md](references/api.md). |

## References (deep dives)

- [references/demos.md](references/demos.md) — Real-world demos: links to [figures4papers](https://github.com/ChenLiu-1996/figures4papers) `figure_*` project folders (bar charts, trends, heatmaps, multi-panel).
- [references/design-theory.md](references/design-theory.md) — Typography, export policy, color theory, layout, bar/line/scatter encoding, reproduction rules.
- [references/api.md](references/api.md) — Full API and conventions to implement.
- [references/common-patterns.md](references/common-patterns.md) — Ultra-wide panels, dedicated legend axis, categorical bars, print-safe colors.
- [references/tutorials.md](references/tutorials.md) — Step-by-step bar chart, multi-panel trend, heatmap.

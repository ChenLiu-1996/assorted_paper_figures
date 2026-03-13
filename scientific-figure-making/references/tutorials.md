# Tutorials: End-to-End Publication Figures

Implement the API described in [api.md](api.md) in your own code (constants, style, plot helpers, finalize). For real-world scripts that follow this style, see [demos.md](demos.md) (links to the [figures4papers](https://github.com/ChenLiu-1996/figures4papers) `figure_*` folders).

---

## Tutorial 1: Bar chart for method comparison

**Goal:** One grouped bar chart comparing 3 methods across 4 metrics, with values above bars and PDF+PNG export.

**Checklist:**

- [ ] Load or define data: categories (metric names), series (one list per method), labels (method names).
- [ ] Call `apply_publication_style` with a style (e.g. `FigureStyle(font_size=24, axes_linewidth=3)` for a large panel).
- [ ] Create figure and one axis; use wide `figsize` if many metrics (e.g. `(16, 5)`).
- [ ] Call `make_grouped_bar(ax, categories, series, labels, ylabel="Score", annotate=True)`.
- [ ] Set y-limits if needed (e.g. `ax.set_ylim(0, 100)`).
- [ ] Call `finalize_figure(fig, "output/comparison", formats=["png", "pdf"], dpi=300)`.

**Example flow** (implement `apply_publication_style`, `make_grouped_bar`, `finalize_figure`, and `PALETTE` per [api.md](api.md)):

```python
import matplotlib.pyplot as plt

# Assume you have implemented: apply_publication_style, FigureStyle, make_grouped_bar, finalize_figure, PALETTE (see api.md)
apply_publication_style(FigureStyle(font_size=24, axes_linewidth=3))
fig, ax = plt.subplots(figsize=(16, 5))

categories = ["Metric A", "Metric B", "Metric C", "Metric D"]
series = [
    [0.92, 0.88, 0.85, 0.90],
    [0.85, 0.82, 0.88, 0.84],
    [0.78, 0.80, 0.82, 0.79],
]
labels = ["Ours", "Baseline X", "Baseline Y"]
make_grouped_bar(
    ax, categories, series, labels,
    ylabel="Score", colors=[PALETTE["blue_main"], PALETTE["green_3"], PALETTE["red_strong"]],
    annotate=True,
)
ax.set_ylim(0.7, 1.0)
finalize_figure(fig, "output/method_comparison", formats=["png", "pdf"], dpi=300)
```

---

## Tutorial 2: Multi-panel trend with shared legend

**Goal:** Two trend panels (e.g. train and validation curves) and a third panel used only for the legend.

**Checklist:**

- [ ] Apply style; create 1×3 subplots (or 2×2 with last axis for legend).
- [ ] Plot trends on the first axes; collect legend handles/labels from one of them.
- [ ] On the last axis, call `ax.set_axis_off()` and add a legend using the collected handles/labels.
- [ ] Set y-limits and titles on data panels.
- [ ] Call `finalize_figure`.

**Example flow** (implement helpers per [api.md](api.md)):

```python
import numpy as np
import matplotlib.pyplot as plt

apply_publication_style(FigureStyle(font_size=14, axes_linewidth=2))
fig, axes = create_subplots(1, 3, figsize=(14, 4))

x = np.linspace(0, 10, 50)
y1 = 0.5 + 0.4 * (1 - np.exp(-x / 3))
y2 = 0.45 + 0.35 * (1 - np.exp(-x / 4))
make_trend(axes[0], x, [y1, y2], ["Model A", "Model B"], ylabel="Loss", xlabel="Step")
axes[0].set_title("Training")
make_trend(axes[1], x, [y1 * 1.1, y2 * 1.05], ["Model A", "Model B"], ylabel="Loss", xlabel="Step")
axes[1].set_title("Validation")

# Legend-only panel
handles, labels = axes[0].get_legend_handles_labels()
axes[2].set_axis_off()
axes[2].legend(handles, labels, loc="center")

finalize_figure(fig, "output/trends", formats=["png", "pdf"], dpi=300)
```

---

## Tutorial 3: Heatmap with labels and colorbar

**Goal:** A correlation or score matrix with row/column labels and a colorbar.

**Checklist:**

- [ ] Apply style; create one axis.
- [ ] Prepare 2D matrix and optional x_labels, y_labels.
- [ ] Call `make_heatmap(ax, matrix, x_labels=..., y_labels=..., cmap="magma", cbar_label="...")`.
- [ ] Call `finalize_figure`.

**Example flow** (implement helpers per [api.md](api.md)):

```python
import numpy as np
import matplotlib.pyplot as plt

apply_publication_style(FigureStyle(font_size=12, axes_linewidth=2))
fig, ax = plt.subplots(figsize=(8, 6))
np.random.seed(42)
matrix = np.random.rand(5, 5)
matrix = (matrix + matrix.T) / 2  # symmetric
labels = [f"F{i+1}" for i in range(5)]
make_heatmap(ax, matrix, x_labels=labels, y_labels=labels, cmap="magma", cbar_label="Correlation")
finalize_figure(fig, "output/heatmap", formats=["png", "pdf"], dpi=300)
```

---

For more patterns (ultra-wide layout, categorical bars, print-safe colors), see [common-patterns.md](common-patterns.md). For design rationale, see [design-theory.md](design-theory.md).

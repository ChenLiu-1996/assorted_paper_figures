# Common Patterns for Publication Figures

These patterns are used across the repository for consistent, publication-ready figures. Apply them when building bar charts, multi-panel layouts, and print-safe graphics.

---

## Ultra-wide aspect for multi-metric panels

For 3–4 metrics or many categories in a single row, use a wide canvas so bars/labels don't crowd vertically.

- **Example:** `figsize=(45, 12)` or `(28, 6)`.
- **Rule:** Width often 3–4× height for comparison bars.
- **Why:** Lets the reader scan left-to-right without squinting; keeps y-axis and legend readable.

---

## Dedicated legend panel

When multiple curves or groups make the legend large, put it in its own subplot so it doesn't cover data.

- **Steps:**
  1. Create one extra subplot (e.g. last in the grid).
  2. Call `ax.set_axis_off()` on that axis.
  3. Build the legend on that axis (e.g. gather handles/labels from other axes and call `ax.legend(handles, labels)` there), or use a single legend that you place in the extra axis.
- **Result:** Data panels stay clean; legend is fully visible.

---

## Categorical bars without x-tick labels

When the x-axis is "method" or "condition" and the legend already identifies them, hide x ticks and rely on the legend/title.

- **Code:** `ax.set_xticks([])` or set tick labels to empty.
- **Use when:** Many methods across multiple metrics; names are in the legend or panel title.

---

## Dynamic y-axis scaling

Tighten y-limits to the relevant range so differences are visible instead of squashed.

- **Idea:** Use something like `data.min() - margin` to `data.max() + margin` (e.g. margin = small fraction of range or one std).
- **Avoid:** Fixed 0–100 when all values are in 85–95; use e.g. 80–100 instead.

---

## Bar edge and hatch for print-safe separation

Bars that differ only by fill can blur in grayscale. Use edges and optional hatching.

- **Edges:** `edgecolor='black'`, `linewidth=1.5`–`3` for clear separation.
- **Hatch:** Same color with different hatch (e.g. `'/'`, `'\\'`, `'.'`) for ablation or subgroups so they remain distinct in print.

---

## Semantic color mapping

Stick to the skill palette so "proposed vs baseline" is consistent across figures.

- **Blues:** Proposed method / key result.
- **Greens:** Improvements / positive variants.
- **Reds:** Baselines / contrasts / alternatives.
- **Neutral:** Reference or background categories.
- **Highlight:** Single callout only.

See [design-theory.md](design-theory.md) for the full palette and [api.md](api.md) for `PALETTE` and `DEFAULT_COLORS`.

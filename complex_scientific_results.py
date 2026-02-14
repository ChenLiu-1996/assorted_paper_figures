from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Import Scientific Figure Pro skill directly from workspace skill path.
sys.path.append(str(Path("skills/scientific-figure-pro/scripts").absolute()))
from scientific_figure_pro import (  # noqa: E402
    FigureStyle,
    PALETTE,
    apply_publication_style,
    create_subplots,
    finalize_figure,
    make_grouped_bar,
    make_heatmap,
    make_sphere_illustration,
    make_trend,
)


def _build_training_curves(epochs: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    curve_a = 0.48 + 0.40 * (1 - np.exp(-epochs / 35.0)) + rng.normal(0, 0.006, size=epochs.size)
    curve_b = 0.45 + 0.36 * (1 - np.exp(-epochs / 40.0)) + rng.normal(0, 0.007, size=epochs.size)
    curve_c = 0.42 + 0.31 * (1 - np.exp(-epochs / 48.0)) + rng.normal(0, 0.008, size=epochs.size)
    return [curve_a, curve_b, curve_c]


def generate_complex_results() -> None:
    rng = np.random.default_rng(7)

    apply_publication_style(FigureStyle(font_size=12, axes_linewidth=1.8))
    fig, axes = create_subplots(2, 2, figsize=(16, 12), constrained_layout=True)

    # Panel A: Trend plot with confidence shadows.
    ax_trend = axes[0]
    epochs = np.arange(1, 121)
    trends = _build_training_curves(epochs, rng)
    make_trend(
        ax_trend,
        x=epochs,
        y_series=trends,
        labels=["Model Alpha", "Model Beta", "Model Gamma"],
        colors=[PALETTE["blue_main"], PALETTE["teal"], PALETTE["red_strong"]],
        xlabel="Epoch",
        ylabel="Validation Accuracy",
        show_shadow=True,
    )
    ax_trend.set_title("A. Training Progress with Confidence Shadows", loc="left", fontweight="bold")
    ax_trend.set_ylim(0.40, 0.92)
    ax_trend.grid(alpha=0.2, linestyle="--")

    # Panel B: Grouped bar chart with annotations.
    ax_bar = axes[1]
    categories = ["Speed", "Accuracy", "Stability", "Memory"]
    series = [
        [86, 91, 84, 79],
        [81, 88, 90, 85],
    ]
    make_grouped_bar(
        ax_bar,
        categories=categories,
        series=series,
        labels=["Model-X", "Model-Y"],
        ylabel="Score",
        colors=[PALETTE["blue_secondary"], PALETTE["green_3"]],
        annotate=True,
    )
    ax_bar.set_title("B. Model Comparison Across Categories", loc="left", fontweight="bold")
    ax_bar.set_ylim(0, 100)

    # Panel C: High-resolution feature correlation heatmap.
    ax_heat = axes[2]
    n_features = 10
    base = rng.normal(size=(700, n_features))
    mix = np.array([
        [1.0, 0.72, 0.10, -0.21, 0.44, 0.15, -0.18, 0.22, 0.38, 0.05],
        [0.72, 1.0, 0.07, -0.26, 0.32, 0.10, -0.22, 0.19, 0.35, 0.08],
        [0.10, 0.07, 1.0, 0.61, -0.03, 0.52, 0.28, -0.18, 0.09, 0.42],
        [-0.21, -0.26, 0.61, 1.0, -0.10, 0.48, 0.33, -0.30, -0.02, 0.37],
        [0.44, 0.32, -0.03, -0.10, 1.0, 0.16, -0.09, 0.58, 0.64, -0.15],
        [0.15, 0.10, 0.52, 0.48, 0.16, 1.0, 0.41, -0.12, 0.05, 0.46],
        [-0.18, -0.22, 0.28, 0.33, -0.09, 0.41, 1.0, -0.25, -0.11, 0.29],
        [0.22, 0.19, -0.18, -0.30, 0.58, -0.12, -0.25, 1.0, 0.47, -0.04],
        [0.38, 0.35, 0.09, -0.02, 0.64, 0.05, -0.11, 0.47, 1.0, 0.12],
        [0.05, 0.08, 0.42, 0.37, -0.15, 0.46, 0.29, -0.04, 0.12, 1.0],
    ])
    transformed = base @ mix
    corr = np.corrcoef(transformed, rowvar=False)
    labels = [f"F{i}" for i in range(1, n_features + 1)]
    make_heatmap(
        ax_heat,
        matrix=corr,
        x_labels=labels,
        y_labels=labels,
        cmap="magma",
        cbar_label="Correlation",
        annotate=False,
    )
    ax_heat.set_title("C. Feature Correlation Matrix", loc="left", fontweight="bold")

    # Panel D: Sphere illustration for theoretical model.
    ax_sphere = axes[3]
    make_sphere_illustration(ax_sphere, light_dir=(-0.55, 0.65, 0.55), resolution=320, alpha=0.95)
    ax_sphere.set_title("D. 3D Sphere Theoretical Model", loc="left", fontweight="bold")

    finalize_figure(
        fig,
        out_path="complex_results",
        formats=["pdf", "png"],
        dpi=450,
        close=True,
        pad=0.03,
    )


if __name__ == "__main__":
    generate_complex_results()
    print("Generated: complex_results.pdf, complex_results.png")

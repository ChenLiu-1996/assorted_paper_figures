import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec


data_ablation = {
    'methods': [
        r'CellSpliceNet',
        r'ViT',
        r'SpliceFinder',
        r'Pangolin',
        r'SpliceTransformer',
        r'SpliceAI',
        r'ESM2',
    ],
    'colors': ['#0F4D92', "#F09F97", "#F1B3AC", "#EFBEB8", "#F0CDC8", "#F3D9D8", '#FCEEED'],
    'metrics': [r'Spearman correlation', r'Pearson correlation', r'R$^2$ score'],
    'result': {
        # r'Spearman correlation': np.array([0.88, 0.81, 0.80, 0.79, 0.77, 0.71, 0.606]),
        # r'Pearson correlation': np.array([0.88, 0.81, 0.80, 0.79, 0.77, 0.72, 0.613]),
        # r'R$^2$ score': np.array([0.77, 0.66, 0.64, 0.62, 0.59, 0.52, 0.369]),
        r'Spearman correlation': np.array([
            [0.88, 0.88, 0.88],  # TODO: update!
            [0.81, 0.81, 0.81],  # TODO: update!
            [0.806, 0.793, 0.794],
            [0.792, 0.785, 0.788],
            [0.765, 0.765, 0.767],
            [0.714, 0.716, 0.706],
            [0.598, 0.625, 0.594],
        ]),
        r'Pearson correlation': np.array([
            [0.88, 0.88, 0.88],  # TODO: update!
            [0.81, 0.81, 0.81],  # TODO: update!
            [0.812, 0.798, 0.801],
            [0.792, 0.785, 0.788],
            [0.765, 0.765, 0.766],
            [0.722, 0.722, 0.714],
            [0.604, 0.631, 0.605],
        ]),
        r'R$^2$ score': np.array([
            [0.77, 0.77, 0.77],  # TODO: update!
            [0.66, 0.66, 0.66],  # TODO: update!
            [0.658, 0.632, 0.642],
            [0.633, 0.615, 0.627],
            [0.585, 0.585, 0.586],
            [0.518, 0.520, 0.507],
            [0.359, 0.384, 0.364],
        ]),
    }
}

def is_dark(color_in_hex, threshold=128):
    color = color_in_hex.lstrip('#')
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)

    luminance = 0.299*r + 0.587*g + 0.114*b
    return luminance < threshold


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'helvetica'
    plt.rcParams['font.size'] = 24
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 3

    fig = plt.figure(figsize=(45, 12))

    gs = gridspec.GridSpec(1, 3)

    for metric_idx, metric_name in enumerate(data_ablation['metrics']):
        ax = fig.add_subplot(gs[metric_idx])

        num_methods = len(data_ablation['methods'])
        bars = ax.bar(
            np.arange(num_methods),
            data_ablation['result'][metric_name].mean(axis=1),
            yerr=data_ablation['result'][metric_name].std(axis=1),
            error_kw={
                'elinewidth': 2,   # thickness of vertical error bar line
                'capthick': 2,     # thickness of caps
                'capsize': 15      # length of caps
            },
            color=data_ablation['colors'],
            label=data_ablation['methods'],
        )

        for i, (bar, value) in enumerate(zip(bars, data_ablation['result'][metric_name].mean(axis=1))):
            textcolor = 'white' if is_dark(data_ablation['colors'][i]) else 'black'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.10,
                f'{value:.2f}', ha='center', va='bottom', fontsize=32, color=textcolor)

        ax.set_ylabel(metric_name, fontsize=54, labelpad=12)
        ymax = np.max(data_ablation['result'][metric_name])
        ax.set_ylim([0.0, ymax + 0.5])
        ax.set_xticks([])
        ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
        ax.tick_params(axis='y', labelsize=36, length=10, width=2)

        ax.legend(bbox_to_anchor=(0.02, 1.08), loc='upper left', fontsize=38, frameon=False, ncols=2, columnspacing=0.6)

    fig.tight_layout(pad=2)

    os.makedirs('./figures/', exist_ok=True)
    fig.savefig('./figures/comparison.png', dpi=300)
    plt.close(fig)

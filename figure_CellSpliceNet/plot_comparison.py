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
    ],
    'colors': ['#0F4D92', "#F1ACA4", "#F1C3BE", "#F3D6D2", "#F5E3E2", '#FCEEED'],
    'metrics': [r'Spearman correlation', r'Pearson correlation', r'R$^2$ score'],
    'result': {
        r'Spearman correlation': np.array([0.88, 0.81, 0.80, 0.79, 0.77, 0.71]),
        r'Pearson correlation': np.array([0.88, 0.81, 0.80, 0.79, 0.77, 0.72]),
        r'R$^2$ score': np.array([0.77, 0.66, 0.64, 0.62, 0.59, 0.52]),
    }
}

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
            data_ablation['result'][metric_name],
            color=data_ablation['colors'],
            label=data_ablation['methods'],
        )

        for i, (bar, value) in enumerate(zip(bars, data_ablation['result'][metric_name])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{value:.2f}', ha='center', va='bottom', fontsize=36)

        ax.set_ylabel(metric_name, fontsize=54, labelpad=12)
        ymin = data_ablation['result'][metric_name].min() - data_ablation['result'][metric_name].std()
        ax.set_ylim([ymin, 1.0])
        ax.set_xticks([])
        tick_spacing = 0.1
        ticks = np.arange(1.0, ymin - tick_spacing, -tick_spacing)[::-1]
        ax.yaxis.set_major_locator(plt.FixedLocator(ticks))
        ax.tick_params(axis='y', labelsize=36)

        ax.legend(bbox_to_anchor=(0.02, 1.08), loc='upper left', fontsize=38, frameon=False, ncols=2, columnspacing=0.6)

    fig.tight_layout(pad=2)

    os.makedirs('./figures/', exist_ok=True)
    fig.savefig('./figures/comparison.png', dpi=300)
    plt.close(fig)

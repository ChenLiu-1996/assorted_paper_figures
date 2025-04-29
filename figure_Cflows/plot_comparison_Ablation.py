import os
import numpy as np
from matplotlib import pyplot as plt


data_comparison_Ablation = {
    'methods': [r'OT-CFM', r'SB-CFM', r'SF2M', r'Cflows (w/o growth + energy)', r'Cflows (w/o growth)', r'Cflows'],
    'colors': ['#AADCA9', '#8BCF8B', '#E9A6A1', '#B8C9E5', '#7097CA', '#3775BA'],
    'metrics': [r'RMSE$\downarrow$', r'MAE$\downarrow$', r'PCC$\uparrow$', r'SCC$\uparrow$'],
    'mean': {
        r'RMSE$\downarrow$': np.array([0.94, 1.05, 0.99, 0.89, 0.75, 0.62]),
        r'MAE$\downarrow$': np.array([0.75, 0.85, 0.78, 0.70, 0.59, 0.48]),
        r'PCC$\uparrow$': np.array([0.53, 0.49, 0.55, 0.58, 0.65, 0.72]),
        r'SCC$\uparrow$': np.array([0.50, 0.47, 0.52, 0.55, 0.68, 0.70]),
    },
    'std': {
        r'RMSE$\downarrow$': np.array([0.08, 0.09, 0.09, 0.07, 0.06, 0.05]),
        r'MAE$\downarrow$': np.array([0.07, 0.08, 0.07, 0.06, 0.05, 0.04]),
        r'PCC$\uparrow$': np.array([0.04, 0.02, 0.01, 0.03, 0.02, 0.02]),
        r'SCC$\uparrow$': np.array([0.03, 0.02, 0.03, 0.03, 0.03, 0.01]),
    },
}


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'helvetica'
    plt.rcParams['font.size'] = 24
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 3

    fig = plt.figure(figsize=(35, 7))

    num_methods = len(data_comparison_Ablation['methods'])
    for metric_idx, metric_name in enumerate(data_comparison_Ablation['metrics']):
        ax = fig.add_subplot(1, 5, metric_idx + 1)

        ax.bar(
            np.arange(num_methods),
            data_comparison_Ablation['mean'][metric_name],
            yerr=data_comparison_Ablation['std'][metric_name],
            capsize=8,
            error_kw={'capthick': 2},
            color=data_comparison_Ablation['colors'],
            label=data_comparison_Ablation['methods'],
        )

        if metric_idx == 0:
            handles, labels = ax.get_legend_handles_labels()

        ax.set_xticks([])
        ax.set_ylabel(data_comparison_Ablation['metrics'][metric_idx], fontsize=36, labelpad=12)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax = fig.add_subplot(1, 5, 5)
    ax.legend(handles, labels, fontsize=30, loc='lower left', frameon=False)
    ax.set_axis_off()

    fig.tight_layout(pad=2)

    os.makedirs('./figures/', exist_ok=True)
    fig.savefig('./figures/figX_comparison_Ablation.png', dpi=300)
    fig.savefig('./figures/figX_comparison_Ablation.pdf', dpi=300)
    plt.close(fig)

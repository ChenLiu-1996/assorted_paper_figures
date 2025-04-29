import os
import numpy as np
from matplotlib import pyplot as plt


data_comparison_Trajectory = {
    'methods': [r'TrajectoryNet', r'OT-CFM', r'SB-CFM', r'BEMIOflow (ours)'],
    'colors': ['#DDF3DE', '#AADCA9', '#8BCF8B', '#3775BA'],
    'metrics': ['PHATE Space RMSE', 'Gene Space RMSE', 'Interpolation EMD'],
    'datasets': ['Bifurcation', 'Cycle', 'Linear'],
    'mean': {
        'PHATE Space RMSE': {
            'Bifurcation': np.array([6.16, 2.98, 2.83, 2.60]) * 1e-3,
            'Cycle': np.array([2.49, 3.79, 1.58, 0.777]) * 1e-3,
            'Linear': np.array([8.08, 5.33, 5.67, 3.40]) * 1e-3,
        },
        'Gene Space RMSE': {
            'Bifurcation': np.array([0.150, 0.0842, 0.0835, 0.0773]),
            'Cycle': np.array([0.151, 0.209, 0.141, 0.118]),
            'Linear': np.array([0.118, 0.0960, 0.0978, 0.0853]),
        },
        'Interpolation EMD': {
            'Bifurcation': np.array([1.07, 0.495, 0.516, 0.465]) * 1e-2,
            'Cycle': np.array([0.710, 0.818, 0.526, 0.465]) * 1e-2,
            'Linear': np.array([1.50, 1.32, 1.28, 0.935]) * 1e-2,
        },
    },
}


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'helvetica'
    plt.rcParams['font.size'] = 24
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 3

    fig = plt.figure(figsize=(30, 7))

    num_methods = len(data_comparison_Trajectory['methods'])
    for metric_idx, metric_name in enumerate(data_comparison_Trajectory['metrics']):
        ax = fig.add_subplot(1, 4, metric_idx + 1)
        xtick_list = []

        for dataset_idx, dataset_name in enumerate(data_comparison_Trajectory['datasets']):

            ax.bar(
                np.arange(num_methods) + dataset_idx * (num_methods + 1),
                data_comparison_Trajectory['mean'][metric_name][dataset_name],
                color=data_comparison_Trajectory['colors'],
                label=data_comparison_Trajectory['methods'],
            )

            xtick_list.append(np.mean(np.arange(num_methods)) + dataset_idx * (num_methods + 1))

            if dataset_idx == 0:
                handles, labels = ax.get_legend_handles_labels()

        ax.set_xticks(xtick_list)
        ax.set_xticklabels(data_comparison_Trajectory['datasets'])
        ax.set_ylabel(data_comparison_Trajectory['metrics'][metric_idx], fontsize=36, labelpad=12)
        ax.set_xlabel('Dataset', fontsize=36, labelpad=12)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax = fig.add_subplot(1, 4, 4)
    ax.legend(handles, labels, fontsize=30, loc='lower left', frameon=False)
    ax.set_axis_off()

    fig.tight_layout(pad=2)

    os.makedirs('./figures/', exist_ok=True)
    fig.savefig('./figures/fig2_comparison_Trajectory.png', dpi=300)
    fig.savefig('./figures/fig2_comparison_Trajectory.pdf', dpi=300)
    plt.close(fig)

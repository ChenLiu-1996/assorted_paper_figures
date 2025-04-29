import os
import numpy as np
from matplotlib import pyplot as plt
from raw_data import data_comparison_IEDB, data_ablation_IEDB, data_comparison_Cancer, data_ablation_Cancer


def decode_ablation(data_dict):
    binary_list = data_dict['ablations']
    component_str = data_dict['components']
    decoded_list = []
    for binary_code in binary_list:
        assert len(binary_code) == len(component_str)
        decoded_str = []
        for i, c in enumerate(binary_code):
            if c == '1':
                decoded_str.append(component_str[i])
        decoded_list.append(' + '.join(decoded_str))
    return decoded_list


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'helvetica'
    plt.rcParams['font.size'] = 24
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 3

    fig = plt.figure(figsize=(28, 6))

    ax = fig.add_subplot(1, 4, 1)
    ax.bar(range(len(data_comparison_IEDB['mean'])),
           data_comparison_IEDB['mean'][:, 0],
           yerr=data_comparison_IEDB['std'][:, 0],
           capsize=5,
           color=data_comparison_IEDB['colors'],
           label=data_comparison_IEDB['methods'])
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xticks([])
    ax.set_ylim([0.5, 0.9])
    ax.set_ylabel(data_comparison_IEDB['metrics'][0], fontsize=32)

    ax = fig.add_subplot(1, 4, 2)
    ax.bar(range(len(data_comparison_IEDB['mean'])),
           data_comparison_IEDB['mean'][:, 1],
           yerr=data_comparison_IEDB['std'][:, 1],
           capsize=5,
           color=data_comparison_IEDB['colors'])
    ax.set_xticks([])
    ax.set_ylim([0.15, 0.75])
    ax.set_ylabel(data_comparison_IEDB['metrics'][1], fontsize=32)

    ax = fig.add_subplot(1, 4, 3)
    ax.bar(range(len(data_comparison_IEDB['mean'])),
           data_comparison_IEDB['mean'][:, 2],
           yerr=data_comparison_IEDB['std'][:, 2],
           capsize=5,
           color=data_comparison_IEDB['colors'])
    ax.set_xticks([])
    ax.set_ylim([0.18, 0.55])
    ax.set_ylabel(data_comparison_IEDB['metrics'][2], fontsize=32)

    ax = fig.add_subplot(1, 4, 4)
    ax.legend(handles, labels)
    ax.set_axis_off()

    fig.tight_layout(pad=2)

    os.makedirs('./figures/', exist_ok=True)
    fig.savefig('./figures/bars_comparison_IEDB.png')
    plt.close(fig)


    fig = plt.figure(figsize=(24, 8))

    ax = fig.add_subplot(1, 3, 1)
    ax.barh(range(len(data_ablation_IEDB['mean'][:, 0])),
            data_ablation_IEDB['mean'][:, 0],
            xerr=data_ablation_IEDB['std'][:, 0],
            color=[(0.215686, 0.458824, 0.729412, alpha) for alpha in np.linspace(0.2, 1.0, 12)],
            ecolor='k',
            capsize=5,
    )

    ax.set_yticks(range(len(data_ablation_IEDB['ablations'])))
    ax.set_yticklabels(decode_ablation(data_ablation_IEDB))
    ax.set_xlim([0.75, 0.9])
    ax.set_xticks([0.75, 0.8, 0.85, 0.9])
    ax.set_xticklabels([0.75, 0.8, 0.85, 0.9])
    ax.set_xlabel(data_ablation_IEDB['metrics'][0], fontsize=32)

    ax = fig.add_subplot(1, 3, 2)
    ax.barh(range(len(data_ablation_IEDB['mean'][:, 1])),
            data_ablation_IEDB['mean'][:, 1],
            xerr=data_ablation_IEDB['std'][:, 1],
            color=[(0.215686, 0.458824, 0.729412, alpha) for alpha in np.linspace(0.2, 1.0, 12)],
            ecolor='k',
            capsize=5,
    )

    ax.set_yticks([])
    ax.set_xlim([0.4, 0.72])
    ax.set_xticks([0.4, 0.5, 0.6, 0.7])
    ax.set_xticklabels([0.4, 0.5, 0.6, 0.7])
    ax.set_xlabel(data_ablation_IEDB['metrics'][1], fontsize=32)

    ax = fig.add_subplot(1, 3, 3)
    ax.barh(range(len(data_ablation_IEDB['mean'][:, 2])),
            data_ablation_IEDB['mean'][:, 2],
            xerr=data_ablation_IEDB['std'][:, 2],
            color=[(0.215686, 0.458824, 0.729412, alpha) for alpha in np.linspace(0.2, 1.0, 12)],
            ecolor='k',
            capsize=5,
    )

    ax.set_yticks([])
    ax.set_xlim([0.4, 0.55])
    ax.set_xticks([0.4, 0.45, 0.5, 0.55])
    ax.set_xticklabels([0.4, 0.45, 0.5, 0.55])
    ax.set_xlabel(data_ablation_IEDB['metrics'][2], fontsize=32)

    fig.tight_layout(pad=2)
    os.makedirs('./figures/', exist_ok=True)
    fig.savefig('./figures/bars_ablation_IEDB.png')
    plt.close(fig)


    fig = plt.figure(figsize=(28, 6))

    ax = fig.add_subplot(1, 4, 1)
    ax.bar(range(len(data_comparison_Cancer['mean'])),
           data_comparison_Cancer['mean'][:, 0],
           yerr=data_comparison_Cancer['std'][:, 0],
           capsize=5,
           color=data_comparison_Cancer['colors'],
           label=data_comparison_IEDB['methods'])
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xticks([])
    ax.set_ylim([0.5, 0.82])
    ax.set_ylabel(data_comparison_Cancer['metrics'][0], fontsize=32)

    ax = fig.add_subplot(1, 4, 2)
    ax.bar(range(len(data_comparison_Cancer['mean'])),
           data_comparison_Cancer['mean'][:, 1],
           yerr=data_comparison_Cancer['std'][:, 1],
           capsize=5,
           color=data_comparison_Cancer['colors'])
    ax.set_xticks([])
    ax.set_ylim([0.16, 0.52])
    ax.set_ylabel(data_comparison_Cancer['metrics'][1], fontsize=32)

    ax = fig.add_subplot(1, 4, 3)
    ax.bar(range(len(data_comparison_Cancer['mean'])),
           data_comparison_Cancer['mean'][:, 2],
           yerr=data_comparison_Cancer['std'][:, 2],
           capsize=5,
           color=data_comparison_Cancer['colors'])
    ax.set_xticks([])
    ax.set_ylim([0.14, 0.44])
    ax.set_ylabel(data_comparison_Cancer['metrics'][2], fontsize=32)

    ax = fig.add_subplot(1, 4, 4)
    ax.legend(handles, labels)
    ax.set_axis_off()

    fig.tight_layout(pad=2)

    os.makedirs('./figures/', exist_ok=True)
    fig.savefig('./figures/bars_comparison_Cancer.png')
    plt.close(fig)


    # fig = plt.figure(figsize=(28, 6))

    # ax = fig.add_subplot(1, 4, 2)
    # # NOTE: Hard-coding for 4 + 4 bars.
    # ax.bar(range(len(data_ablation_Cancer['mean'])),
    #        data_ablation_Cancer['mean'][:, 0],
    #        yerr=data_ablation_Cancer['std'][:, 0],
    #        capsize=5,
    #        hatch=['/'] * 4 + [''] * 4,
    #        color=[(0.215686, 0.458824, 0.729412, alpha) for alpha in [0.25, 0.5, 0.75, 1.0]] * 2,
    #        label=[None, None, 'No Transfer Learning', None, None, None, 'Transfer Learning', None])
    # # for i in range(len(data_comparison_Cancer['mean']) - 1):
    # #     ax.hlines(data_comparison_Cancer['mean'][i, 0],
    # #               xmin=-1, xmax=len(data_comparison_Cancer['mean'])-1,
    # #               linestyles='-', colors=data_comparison_Cancer['colors'][i],
    # #               linewidth=2, label=data_comparison_Cancer['methods'][i])
    # handles, labels = ax.get_legend_handles_labels()
    # ax.set_xticks(range(len(data_ablation_Cancer['coeffs'])))
    # ax.set_xticklabels(['    ' + format(coeff, '.0e') if coeff != 0 else coeff
    #                     for (_, coeff) in data_ablation_Cancer['coeffs']], rotation=90)
    # ax.set_xlabel(r'Contrastive Coefficient $\lambda_\text{CW}$')
    # # ax.set_xlim([-0.8, len(data_comparison_Cancer['mean'])-1.2])
    # ax.set_ylim([0.65, 0.82])
    # ax.set_ylabel(data_ablation_Cancer['metrics'][0], fontsize=32)

    # ax = fig.add_subplot(1, 4, 3)
    # ax.bar(range(len(data_ablation_Cancer['mean'])),
    #        data_ablation_Cancer['mean'][:, 1],
    #        yerr=data_ablation_Cancer['std'][:, 1],
    #        capsize=5,
    #        hatch=['/'] * 4 + [''] * 4,
    #        color=[(0.215686, 0.458824, 0.729412, alpha) for alpha in [0.25, 0.5, 0.75, 1.0]] * 2)
    # # for i in range(len(data_comparison_Cancer['mean']) - 1):
    # #     ax.hlines(data_comparison_Cancer['mean'][i, 1],
    # #               xmin=-1, xmax=len(data_comparison_Cancer['mean'])-1,
    # #               linestyles='-', colors=data_comparison_Cancer['colors'][i],
    # #               linewidth=3)
    # ax.set_xticks(range(len(data_ablation_Cancer['coeffs'])))
    # ax.set_xticklabels(['    ' + format(coeff, '.0e') if coeff != 0 else coeff
    #                     for (_, coeff) in data_ablation_Cancer['coeffs']], rotation=90)
    # ax.set_xlabel(r'Contrastive Coefficient $\lambda_\text{CW}$')
    # # ax.set_xlim([-0.8, len(data_comparison_Cancer['mean'])-1.2])
    # ax.set_ylim([0.28, 0.52])
    # ax.set_ylabel(data_ablation_Cancer['metrics'][1], fontsize=32)

    # ax = fig.add_subplot(1, 4, 4)
    # ax.bar(range(len(data_ablation_Cancer['mean'])),
    #        data_ablation_Cancer['mean'][:, 2],
    #        yerr=data_ablation_Cancer['std'][:, 2],
    #        capsize=5,
    #        hatch=['/'] * 4 + [''] * 4,
    #        color=[(0.215686, 0.458824, 0.729412, alpha) for alpha in [0.25, 0.5, 0.75, 1.0]] * 2)
    # # for i in range(len(data_comparison_Cancer['mean']) - 1):
    # #     ax.hlines(data_comparison_Cancer['mean'][i, 2],
    # #               xmin=-1, xmax=len(data_comparison_Cancer['mean'])-1,
    # #               linestyles='-', colors=data_comparison_Cancer['colors'][i],
    # #               linewidth=3)
    # ax.set_xticks(range(len(data_ablation_Cancer['coeffs'])))
    # ax.set_xticklabels(['    ' + format(coeff, '.0e') if coeff != 0 else coeff
    #                     for (_, coeff) in data_ablation_Cancer['coeffs']], rotation=90)
    # ax.set_xlabel(r'Contrastive Coefficient $\lambda_\text{CW}$')
    # # ax.set_xlim([-0.8, len(data_comparison_Cancer['mean'])-1.2])
    # ax.set_ylim([0.26, 0.44])
    # ax.set_ylabel(data_ablation_Cancer['metrics'][2], fontsize=32)

    # ax = fig.add_subplot(1, 4, 1)
    # ax.bar(range(len(data_ablation_Cancer['mean'])),
    #        data_ablation_Cancer['mean'].mean(-1),
    #        capsize=5,
    #        hatch=['/'] * 4 + [''] * 4,
    #        color=[(0.215686, 0.458824, 0.729412, alpha) for alpha in [0.25, 0.5, 0.75, 1.0]] * 2)
    # ax.set_xticks(range(len(data_ablation_Cancer['coeffs'])))
    # ax.set_xticklabels(['    ' + format(coeff, '.0e') if coeff != 0 else coeff
    #                     for (_, coeff) in data_ablation_Cancer['coeffs']], rotation=90)
    # ax.set_xlabel(r'Contrastive Coefficient $\lambda_\text{CW}$')
    # # ax.set_xlim([-0.8, len(data_comparison_Cancer['mean'])-1.2])
    # ax.set_ylim([0.46, 0.52])
    # ax.set_ylabel(r'$\frac{\text{AUROC} + \text{AUPRC} + \text{Mean PPVn}}{3}$', fontsize=24)

    # # ax = fig.add_subplot(1, 4, 4)
    # # ax.legend(handles, labels)
    # # ax.set_axis_off()

    # fig.tight_layout(pad=2)

    fig = plt.figure(figsize=(28, 6))

    items_shown = [6, 4, 0] # transfer + contrastive, transfer, none

    ax = fig.add_subplot(1, 4, 1)
    ax.bar(range(len(items_shown)),
           data_ablation_Cancer['mean'][:, 0][items_shown],
           yerr=data_ablation_Cancer['std'][:, 0][items_shown],
           capsize=5,
           color=[(0.215686, 0.458824, 0.729412, alpha) for alpha in [1.0, 0.7, 0.4]],
           label=['ImmunoStruct', 'No Contrastive Learning',
                  'No Contrastive Learning &\nNo Transfer Learning'])
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xticks([])
    ax.set_ylim([0.68, 0.80])
    ax.set_ylabel(data_ablation_Cancer['metrics'][0], fontsize=32)

    ax = fig.add_subplot(1, 4, 2)
    ax.bar(range(len(items_shown)),
           data_ablation_Cancer['mean'][:, 1][items_shown],
           yerr=data_ablation_Cancer['std'][:, 1][items_shown],
           capsize=5,
           color=[(0.215686, 0.458824, 0.729412, alpha) for alpha in [1.0, 0.7, 0.4]])
    ax.set_xticks([])
    ax.set_ylim([0.30, 0.52])
    ax.set_ylabel(data_ablation_Cancer['metrics'][1], fontsize=32)

    ax = fig.add_subplot(1, 4, 3)
    ax.bar(range(len(items_shown)),
           data_ablation_Cancer['mean'][:, 2][items_shown],
           yerr=data_ablation_Cancer['std'][:, 2][items_shown],
           capsize=5,
           color=[(0.215686, 0.458824, 0.729412, alpha) for alpha in [1.0, 0.7, 0.4]])
    ax.set_xticks([])
    ax.set_ylim([0.29, 0.43])
    ax.set_ylabel(data_ablation_Cancer['metrics'][2], fontsize=32)

    ax = fig.add_subplot(1, 4, 4)
    ax.legend(handles, labels)
    ax.set_axis_off()

    fig.tight_layout(pad=2)
    os.makedirs('./figures/', exist_ok=True)
    fig.savefig('./figures/bars_ablation_Cancer.png')
    plt.close(fig)

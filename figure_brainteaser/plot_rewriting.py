import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec


data_rewriting_math = {
    'methods': [r'DeepSeek R1 Distill Llama 70B',
                r'deepseek-reasoner (Deepseek-R1)',
                r'OpenAI o3'],
    'colors': ['#8BCF8B', '#E9A6A1', '#3775BA'],
    'hatch_styles': ['', '|', '\\', '/', '-'],
    'fig1': ['Before rewriting', 'After rewriting'],
    'fig2': [r'correct $\rightarrow$ incorrect', r'incorrect $\rightarrow$ correct', 'same result'],
    'result': {
        'Before rewriting': np.array([7, 15, 17]) / 30,
        'After rewriting': np.array([10, 19, 22]) / 30,
        r'correct $\rightarrow$ incorrect': np.array([0, 2, 1]) / 30,
        r'incorrect $\rightarrow$ correct': np.array([3, 6, 6]) / 30,
        'same result': np.array([27, 22, 23]) / 30,
    },
}

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'helvetica'
    plt.rcParams['font.size'] = 24
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 3

    fig = plt.figure(figsize=(24, 12))

    gs = gridspec.GridSpec(2, 2)
    num_methods = len(data_rewriting_math['methods'])

    ax = fig.add_subplot(gs[0])
    width = 0.3
    for category_idx, category in enumerate(data_rewriting_math['fig1']):
        ax.bar(np.arange(num_methods) + width * category_idx * 1.1,
               data_rewriting_math['result'][category],
               width=width,
               label=category,
               color=data_rewriting_math['colors'],
               edgecolor='black',
               linewidth=2,
               hatch=data_rewriting_math['hatch_styles'][category_idx])
    ax.set_title('Correctness', fontsize=36, pad=0)
    ax.set_ylabel('Probability', fontsize=30, labelpad=12)
    ax.set_ylim([0, 1.01])
    ax.set_xticks([])

    ax = fig.add_subplot(gs[1])
    width = 0.25
    for category_idx, category in enumerate(data_rewriting_math['fig2']):
        ax.bar(np.arange(num_methods) + width * category_idx * 1.1,
               data_rewriting_math['result'][category],
               width=width,
               label=category,
               color=data_rewriting_math['colors'],
               edgecolor='black',
               linewidth=2,
               hatch=data_rewriting_math['hatch_styles'][category_idx + 2])
    ax.set_title('Change in Result', fontsize=36, pad=0)
    ax.set_ylabel('Probability', fontsize=30, labelpad=12)
    ax.set_ylim([0, 1.01])
    ax.set_xticks([])

    ax = fig.add_subplot(gs[2])
    bar = ax.bar(
        np.arange(num_methods),
        np.ones_like(np.arange(num_methods)),
        color=data_rewriting_math['colors'],
        edgecolor='black',
        linewidth=2,
        label=data_rewriting_math['methods'],
        hatch='',
    )
    handles, labels = ax.get_legend_handles_labels()
    for b in bar:
        b.remove()
    ax.legend(handles, labels, fontsize=30, loc='center', frameon=False)
    ax.set_axis_off()

    ax = fig.add_subplot(gs[3])
    subtypes = data_rewriting_math['fig1'] + data_rewriting_math['fig2']
    bar = ax.bar(
        np.arange(len(subtypes)),
        np.ones_like(np.arange(len(subtypes))),
        color='white',
        edgecolor='black',
        linewidth=2,
        label=subtypes,
        hatch=data_rewriting_math['hatch_styles'],
    )
    handles, labels = ax.get_legend_handles_labels()
    for b in bar:
        b.remove()
    ax.legend(handles, labels, fontsize=30, loc='center', frameon=False)
    ax.set_axis_off()

    fig.tight_layout(pad=2)

    os.makedirs('./figures/', exist_ok=True)
    fig.savefig('./figures/rewriting.png', dpi=300)
    plt.close(fig)

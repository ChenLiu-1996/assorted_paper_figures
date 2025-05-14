import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from matplotlib import patheffects as path_effects


data_brute_force_math = {
    'methods': [r'DeepSeek R1 Distill Llama 70B',
                r'deepseek-reasoner (Deepseek-R1)',
                r'OpenAI o3'],
    'prompts': ['Basic Prompt', 'Math Prompt', 'Hint Prompt', 'Math + Hint'],
    'colors': ['#8BCF8B', '#E9A6A1', '#3775BA'],
    'subtypes': [r'$\bf{Only}$ $\bf{Model}$ brute force',
                 r'$\bf{Only}$ $\bf{Human}$ brute force',
                 r'$\bf{Neither}$ brute force',
                 r'$\bf{Both}$ brute force'],
    'hatch_styles': ['/', '\\', '', 'x'],
    'result': {
        'Basic Prompt': np.array([[30.4, 1.6, 59.6, 8.4],
                                  [14.8, 3.6, 75.2, 6.4],
                                  [11.4, 4.5, 78.5, 5.7]]) / 100,
        'Math Prompt': np.array([[30.0, 3.2, 60.0, 6.8],
                                 [11.6, 2.0, 78.4, 8.0],
                                 [5.2, 6.0, 84.7, 4.0]]) / 100,
        'Hint Prompt': np.array([[28.0, 3.2, 62.0, 6.8],
                                 [17.2, 2.8, 72.8, 7.2],
                                 [8.4, 7.2, 81.6, 2.8]]) / 100,
        'Math + Hint': np.array([[28.0, 3.2, 62.0, 6.8],
                                 [12.4, 3.2, 77.6, 6.8],
                                 [4.6, 6.3, 85.4, 3.8]]) / 100,
        },
}

data_brute_force_logic = {
    'methods': [r'DeepSeek R1 Distill Llama 70B',
                r'deepseek-reasoner (Deepseek-R1)',
                r'OpenAI o3'],
    'prompts': ['Basic Prompt', 'Math Prompt', 'Hint Prompt', 'Math + Hint'],
    'colors': ['#8BCF8B', '#E9A6A1', '#3775BA'],
    'subtypes': [r'$\bf{Only}$ $\bf{Model}$ brute force',
                 r'$\bf{Only}$ $\bf{Human}$ brute force',
                 r'$\bf{Neither}$ brute force',
                 r'$\bf{Both}$ brute force'],
    'hatch_styles': ['/', '\\', '', 'x'],
    'result': {
        'Basic Prompt': np.array([[30.8, 6.0, 59.2, 4.0],
                                  [16.5, 7.6, 74.4, 2.0],
                                  [10.3, 7.5, 81.3, 0.9]]) / 100,
        'Math Prompt': np.array([[28.8, 6.0, 61.2, 4.0],
                                 [15.6, 7.6, 74.4, 2.4],
                                 [6.4, 8.2, 85.0, 0.5]]) / 100,
        'Hint Prompt': np.array([[29.2, 7.2, 60.8, 2.8],
                                 [12.8, 7.2, 77.2, 2.8],
                                 [7.6, 10.0, 82.4, 0.0]]) / 100,
        'Math + Hint': np.array([[28.0, 7.6, 62.0, 2.4],
                                 [10.8, 8.4, 79.2, 1.6],
                                 [9.8, 8.4, 80.8, 1.9]]) / 100,
        },
}

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'helvetica'
    plt.rcParams['font.size'] = 24
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 3

    fig = plt.figure(figsize=(36, 12))

    gs = gridspec.GridSpec(2, 5)

    for prompt_idx, prompt_name in enumerate(data_brute_force_math['prompts']):
        ax = fig.add_subplot(gs[prompt_idx])
        num_methods = len(data_brute_force_math['methods'])
        bars = ax.bar(
            np.arange(num_methods),
            data_brute_force_math['result'][prompt_name][:, 0],
            color=data_brute_force_math['colors'],
            label=data_brute_force_math['methods'],
            hatch=data_brute_force_math['hatch_styles'][0],
            edgecolor='black',
            linewidth=2,
        )

        for bar in bars:
            height = bar.get_height()
            text = ax.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,
                f'{height:.3f}',
                ha='center',
                va='center',
                color='#FFD700',
                fontsize=20,
                path_effects=[
                    path_effects.Stroke(linewidth=4, foreground='black'),
                    path_effects.Normal()
                ]
            )

        for subtype_idx in range(1, len(data_brute_force_math['subtypes'])):
            ax.bar(
                np.arange(num_methods),
                data_brute_force_math['result'][prompt_name][:, subtype_idx],
                color=data_brute_force_math['colors'],
                label=data_brute_force_math['methods'],
                hatch=data_brute_force_math['hatch_styles'][subtype_idx],
                bottom=np.cumsum(data_brute_force_math['result'][prompt_name], axis=1)[:, subtype_idx - 1],
                edgecolor='black',
                linewidth=2,
                alpha=0.8,
            )

        ax.set_title(data_brute_force_math['prompts'][prompt_idx], fontsize=36, pad=36)
        ax.set_ylabel('Probability', fontsize=30, labelpad=12)
        ax.set_ylim([0, 1.01])
        ax.set_xticks([])

    ax = fig.add_subplot(gs[4])
    bar = ax.bar(
        np.arange(num_methods),
        np.ones_like(np.arange(num_methods)),
        color=data_brute_force_math['colors'],
        label=data_brute_force_math['methods'],
        hatch='',
        edgecolor='black',
        linewidth=3,
    )
    handles, labels = ax.get_legend_handles_labels()
    for b in bar:
        b.remove()
    ax.legend(handles, labels, fontsize=30, loc='center', frameon=False)
    ax.set_axis_off()

    for prompt_idx, prompt_name in enumerate(data_brute_force_logic['prompts']):
        ax = fig.add_subplot(gs[prompt_idx + 5])
        num_methods = len(data_brute_force_logic['methods'])
        bars = ax.bar(
            np.arange(num_methods),
            data_brute_force_logic['result'][prompt_name][:, 0],
            color=data_brute_force_logic['colors'],
            label=data_brute_force_logic['methods'],
            hatch=data_brute_force_logic['hatch_styles'][0],
            edgecolor='black',
            linewidth=2,
        )

        for bar in bars:
            height = bar.get_height()
            text = ax.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,
                f'{height:.3f}',
                ha='center',
                va='center',
                color='#FFD700',
                fontsize=20,
                path_effects=[
                    path_effects.Stroke(linewidth=4, foreground='black'),
                    path_effects.Normal()
                ]
            )

        for subtype_idx in range(1, len(data_brute_force_logic['subtypes'])):
            ax.bar(
                np.arange(num_methods),
                data_brute_force_logic['result'][prompt_name][:, subtype_idx],
                color=data_brute_force_logic['colors'],
                label=data_brute_force_logic['methods'],
                hatch=data_brute_force_logic['hatch_styles'][subtype_idx],
                bottom=np.cumsum(data_brute_force_logic['result'][prompt_name], axis=1)[:, subtype_idx - 1],
                edgecolor='black',
                linewidth=2,
                alpha=0.8,
            )

        ax.set_title(data_brute_force_logic['prompts'][prompt_idx], fontsize=36, pad=36)
        ax.set_ylabel('Probability', fontsize=30, labelpad=12)
        ax.set_ylim([0, 1.01])
        ax.set_xticks([])

    ax = fig.add_subplot(gs[9])
    num_subtypes = len(data_brute_force_math['subtypes'])
    bar = ax.bar(
        np.arange(num_subtypes),
        np.ones_like(np.arange(num_subtypes)),
        color='white',
        label=data_brute_force_math['subtypes'],
        hatch=data_brute_force_math['hatch_styles'],
        edgecolor='black',
        linewidth=3,
    )
    handles, labels = ax.get_legend_handles_labels()
    for b in bar:
        b.remove()
    ax.legend(handles, labels, fontsize=30, loc='center', frameon=False)
    ax.set_axis_off()

    fig.tight_layout(pad=2)

    os.makedirs('./figures/', exist_ok=True)
    fig.savefig('./figures/brute_force.png', dpi=300)
    plt.close(fig)

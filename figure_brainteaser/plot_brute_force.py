import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from matplotlib import patheffects as path_effects


data_brute_force_math = {
    'methods': [
        r'DeepSeek R1 Distill Qwen 1.5B',
        r'DeepSeek R1 Distill Qwen 14B',
        r'DeepSeek R1 Distill Llama 70B',
        r'deepseek-chat (Deepseek-V3)',
        r'deepseek-reasoner (Deepseek-R1)',
        r'gemini-2.5-flash-preview-04-17',
        r'OpenAI o3'
    ],
    'colors': ['#DDF3DE', '#AADCA9', '#8BCF8B', '#F6CFCB', '#E9A6A1', '#FFF6CC', '#3775BA'],
    'prompts': ['CoT Prompt', 'Math Prompt', 'Hint Prompt', 'Math + Hint'],
    'subtypes': [r'$\bf{Only}$ $\bf{Model}$ brute force',
                 r'$\bf{Only}$ $\bf{Human}$ brute force',
                 r'$\bf{Neither}$ brute force',
                 r'$\bf{Both}$ brute force'],
    'hatch_styles': ['/', '\\', '', 'x'],
    'result': {
        'CoT Prompt': np.array([[26.8, 3.6, 60.0, 9.6],
                                [27.6, 3.2, 59.2, 10.0],
                                [24.4, 4.0, 62.4, 9.2],
                                [31.2, 3.6, 55.6, 9.6],
                                [14.0, 7.2, 72.8, 6.0],
                                [16.9, 5.9, 70.0, 7.2],
                                [9.5, 7.5, 79.4, 3.5]]) / 100,
        'Math Prompt': np.array([[27.2, 4.8, 59.6, 8.4],
                                 [25.6, 4.8, 61.2, 8.4],
                                 [24.4, 4.8, 62.4, 8.4],
                                 [28.4, 3.2, 58.4, 10.0],
                                 [10.0, 5.6, 76.7, 7.6],
                                 [12.6, 5.2, 74.8, 7.4],
                                 [4.2, 7.9, 85.7, 2.1]]) / 100,
        'Hint Prompt': np.array([[29.6, 4.4, 57.2, 8.8],
                                 [27.6, 2.8, 59.2, 10.4],
                                 [20.4, 5.2, 66.4, 8.0],
                                 [28.0, 3.2, 58.8, 10.0],
                                 [14.0, 6.0, 72.8, 7.2],
                                 [12.4, 5.1, 74.4, 8.1],
                                 [6.8, 6.8, 83.2, 3.1]]) / 100,
        'Math + Hint': np.array([[26.4, 4.0, 60.4, 9.2],
                                 [27.6, 3.2, 59.2, 10.0],
                                 [24.0, 4.8, 62.8, 8.4],
                                 [25.2, 2.8, 61.6, 10.4],
                                 [8.0, 6.8, 78.8, 6.4],
                                 [10.0, 6.1, 76.4, 7.4],
                                 [3.7, 6.4, 86.7, 3.2]]) / 100,
    },
}

data_brute_force_logic = {
    'methods': [
        r'DeepSeek R1 Distill Qwen 1.5B',
        r'DeepSeek R1 Distill Qwen 14B',
        r'DeepSeek R1 Distill Llama 70B',
        r'deepseek-chat (Deepseek-V3)',
        r'deepseek-reasoner (Deepseek-R1)',
        r'gemini-2.5-flash-preview-04-17',
        r'OpenAI o3'
    ],
    'colors': ['#DDF3DE', '#AADCA9', '#8BCF8B', '#F6CFCB', '#E9A6A1', '#FFF6CC', '#3775BA'],
    'prompts': ['CoT Prompt', 'Math Prompt', 'Hint Prompt', 'Math + Hint'],
    'subtypes': [r'$\bf{Only}$ $\bf{Model}$ brute force',
                 r'$\bf{Only}$ $\bf{Human}$ brute force',
                 r'$\bf{Neither}$ brute force',
                 r'$\bf{Both}$ brute force'],
    'hatch_styles': ['/', '\\', '', 'x'],
    'result': {
        'CoT Prompt': np.array([[18.0, 5.2, 72.0, 4.8],
                                [30.8, 3.2, 59.2, 6.8],
                                [23.2, 2.8, 66.8, 7.2],
                                [32.4, 3.6, 57.6, 6.4],
                                [13.7, 6.8, 76.7, 2.8],
                                [15.7, 3.2, 75.6, 5.5],
                                [15.9, 6.5, 75.6, 2.0]]) / 100,
        'Math Prompt': np.array([[20.0, 4.4, 70.0, 5.6],
                                 [31.2, 4.0, 58.8, 6.0],
                                 [24.4, 4.0, 65.6, 6.0],
                                 [33.6, 3.6, 56.4, 6.4],
                                 [12.4, 7.6, 77.6, 2.4],
                                 [15.2, 5.5, 75.7, 3.7],
                                 [10.5, 6.2, 80.9, 2.4]]) / 100,
        'Hint Prompt': np.array([[15.6, 4.8, 74.4, 5.2],
                                 [30.4, 3.2, 59.6, 6.8],
                                 [25.2, 2.8, 64.8, 7.2],
                                 [27.6, 2.8, 62.4, 7.2],
                                 [9.6, 6.0, 80.7, 3.6],
                                 [14.3, 4.5, 75.8, 5.4],
                                 [6.9, 5.9, 84.8, 2.5]]) / 100,
        'Math + Hint': np.array([[20.0, 3.2, 70.0, 6.8],
                                 [32.0, 4.8, 58.0, 5.2],
                                 [19.6, 4.0, 70.4, 6.0],
                                 [26.0, 4.0, 64.0, 6.0],
                                 [8.8, 5.6, 81.5, 4.0],
                                 [13.0, 5.3, 76.8, 4.8],
                                 [5.1, 7.4, 86.5, 0.9]]) / 100,
    },
}

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'helvetica'
    plt.rcParams['font.size'] = 24
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 3

    fig = plt.figure(figsize=(52, 12))

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

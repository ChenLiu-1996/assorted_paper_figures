import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec


data_math_by_category = {
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
    'subtypes': ['Standard',
                 'Nonstandard',
                 'Heuristic'],
    'result': {
        'Standard': np.array([0.253623188, 0.5, 0.5, 0.594202899, 0.688405797, 0.710144928, 0.789855073]),
        'Geometry': np.array([0.125, 0.416666667, 0.375, 0.291666667, 0.375, 0.541666667, 0.666666667]),
        'Number Theory': np.array([0.441176471, 0.558823529, 0.529411765, 0.676470588, 0.823529412, 0.735294118, 0.852941177]),
        'Combinatorics': np.array([0.25, 0.416666667, 0.416666667, 0.625, 0.666666667, 0.625, 0.75]),
        'Algebra': np.array([0.196428571, 0.535714286, 0.571428571, 0.660714286, 0.75, 0.803571429, 0.821428571]),
        'Nonstandard': np.array([0.103448276, 0.482758621, 0.431034483, 0.620689655, 0.74137931, 0.672413793, 0.827586207]),
        'Logic': np.array([0.034482759, 0.413793103, 0.448275862, 0.517241379, 0.620689655, 0.517241379, 0.75862069]),
        'Special Number': np.array([0.172413793, 0.551724138, 0.413793103, 0.724137931, 0.862068966, 0.827586207, 0.896551724]),
        'Heuristic': np.array([0, 0.260869565, 0.173913044, 0.413043478, 0.673913044, 0.47826087, 0.804347826]),
        'Pattern': np.array([0, 0.214285714, 0.178571429, 0.357142857, 0.642857143, 0.428571429, 0.75]),
        'Arithmetic': np.array([0, 0.333333333, 0.166666667, 0.5, 0.722222222, 0.555555556, 0.888888889]),
        },
}

data_logic_by_category = {
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
    'subtypes': ['Simple/large',
                 'Complex/small',
                 'Math-like',
                 'Heuristic'],
    'result': {
        'Simple/large': np.array([0.042105263, 0.178947368, 0.189473684, 0.389473684, 0.410526316, 0.515789474, 0.694736842]),
        '0D': np.array([0.068965517, 0.172413793, 0.206896552, 0.379310345, 0.448275862, 0.517241379, 0.689655172]),
        '1D': np.array([0, 0.230769231, 0.230769231, 0.615384615, 0.538461539, 0.538461539, 0.692307692]),
        '2D': np.array([0, 0.045454545, 0.090909091, 0.272727273, 0.181818182, 0.363636364, 0.454545455]),
        'Number': np.array([0.058823529, 0.470588235, 0.411764706, 0.588235294, 0.823529412, 0.941176471, 1]),
        'Clusters': np.array([0, 0, 0, 0.125, 0, 0.25, 0.875]),
        'Tree': np.array([0.166666667, 0, 0, 0.166666667, 0.166666667, 0.166666667, 0.5]),
        'Complex/small': np.array([0, 0.433333333, 0.4, 0.466666667, 0.566666667, 0.833333333, 0.866666667]),
        'Liars': np.array([0, 0.411764706, 0.411764706, 0.411764706, 0.705882353, 0.882352941, 0.882352941]),
        'Communication': np.array([0, 0.5, 0, 0.25, 0, 0.5, 0.5]),
        'Compound': np.array([0, 0.444444444, 0.555555556, 0.666666667, 0.555555556, 0.888888889, 1]),
        'Math-like': np.array([0.085714286, 0.328571429, 0.371428571, 0.514285714, 0.542857143, 0.542857143, 0.714285714]),
        'Algorithm': np.array([0.078947368, 0.315789474, 0.368421053, 0.473684211, 0.447368421, 0.473684211, 0.710526316]),
        'Math': np.array([0.09375, 0.34375, 0.375, 0.5625, 0.65625, 0.625, 0.71875]),
        'Heuristic': np.array([0, 0.12195122, 0.097560976, 0.317073171, 0.365853659, 0.268292683, 0.658536585]),
        'Pattern': np.array([0, 0.153846154, 0.115384615, 0.230769231, 0.346153846, 0.307692308, 0.576923077]),
        'Linguistic': np.array([0, 0.066666667, 0.066666667, 0.466666667, 0.4, 0.2, 0.8]),        },
}


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'helvetica'
    plt.rcParams['font.size'] = 24
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 3

    fig = plt.figure(figsize=(36, 12))

    gs = gridspec.GridSpec(2, 4)

    for subtype_idx, subtype_name in enumerate(data_math_by_category['subtypes']):
        ax = fig.add_subplot(gs[subtype_idx])
        num_methods = len(data_math_by_category['methods'])
        ax.bar(
            np.arange(num_methods),
            data_math_by_category['result'][subtype_name],
            color=data_math_by_category['colors'],
            label=data_math_by_category['methods'],
        )

        ax.set_title(data_math_by_category['subtypes'][subtype_idx], fontsize=36, pad=36)
        ax.set_ylabel('Probability', fontsize=30, labelpad=12)
        ax.set_ylim([0, 1])
        ax.set_xticks([])

    ax = fig.add_subplot(gs[3])
    bar = ax.bar(
        np.arange(num_methods),
        np.ones_like(np.arange(num_methods)),
        color=data_math_by_category['colors'],
        label=data_math_by_category['methods'],
        hatch='',
    )
    handles, labels = ax.get_legend_handles_labels()
    for b in bar:
        b.remove()
    ax.legend(handles, labels, fontsize=28, loc='center', frameon=False)
    ax.set_axis_off()

    for subtype_idx, subtype_name in enumerate(data_logic_by_category['subtypes']):
        ax = fig.add_subplot(gs[4 + subtype_idx])
        num_methods = len(data_logic_by_category['methods'])
        ax.bar(
            np.arange(num_methods),
            data_logic_by_category['result'][subtype_name],
            color=data_logic_by_category['colors'],
            label=data_logic_by_category['methods'],
        )

        ax.set_title(data_logic_by_category['subtypes'][subtype_idx], fontsize=36, pad=36)
        ax.set_ylabel('Probability', fontsize=30, labelpad=12)
        ax.set_ylim([0, 1])
        ax.set_xticks([])


    fig.tight_layout(pad=2)

    os.makedirs('./figures/', exist_ok=True)
    fig.savefig('./figures/correctness_by_category.png', dpi=300)
    plt.close(fig)

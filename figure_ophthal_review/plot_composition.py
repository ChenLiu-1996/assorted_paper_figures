import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


DATA = {
    'clinical_stage': [
        'Benchmark\nEvaluation', 'Expert\nEvaluation', 'Retrospective\nClinical Validation',
        'Prospective\nPilot Study', 'Full\nClinical Trial',
    ],
    'pub_by_category':
        {
            'Clinical Workflow': {
                'Screening or Diagnosis': [10, 7, 10, 3, 0],
                'Report Generation': [2, 3, 3, 0, 0],
                'Treatment Planning or\nRecommendation': [2, 3, 3, 2, 0],
            },
            'Patient Support': {
                'Patient Question Answering': [5, 13, 1, 1, 0],
                'After Visit or Discharge\nSummary Generation': [0, 2, 0, 0, 0],
                'Consultation or Interview': [0, 1, 1, 0, 0],
                'Patient Education\nMaterial Generation': [1, 6, 0, 0, 0],
                'Physician Recommendation': [0, 1, 0, 0, 0],
            },
            'Education and Training': {
                'Exam Taking': [19, 5, 0, 0, 0],
                'Medical Education and\nLearning Support': [3, 5, 0, 0, 0],
            }
        }
}

def plot_heatmap(fig_name: str):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'helvetica'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 2

    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    fig = plt.figure(figsize=(12, 10))

    ax = fig.add_subplot(1, 1, 1)
    category_count_arr, category_arr = [], []
    subcategory_arr = []
    value_arr = []
    for category in DATA['pub_by_category']:
        subcategories = DATA['pub_by_category'][category].keys()
        category_count_arr.append(len(subcategories))
        category_arr.append(category)
        for subcategory in subcategories:
            value = DATA['pub_by_category'][category][subcategory]
            subcategory_arr.append(subcategory)
            value_arr.append(value)
    value_arr = np.stack(value_arr, axis=0)
    for loc, item in enumerate(subcategory_arr):
        item += '\n' + rf'($n={value_arr[loc, :].sum()}$)'
        subcategory_arr[loc] = item
    stage_arr = DATA['clinical_stage']
    for loc, item in enumerate(stage_arr):
        item += '\n' + rf'($n={value_arr[:, loc].sum()}$)'
        stage_arr[loc] = item

    sns.heatmap(value_arr, annot=True, fmt='d', cmap='viridis',
                linewidths=1, linecolor='white', ax=ax, cbar=False)
    ax.set_yticks(np.arange(len(subcategory_arr)) + 0.5)
    ax.set_yticklabels(subcategory_arr, rotation=0)
    ax.set_xticks(np.arange(len(stage_arr)) + 0.5)
    ax.set_xticklabels(stage_arr, rotation=0)

    fig.tight_layout(pad=2)
    fig.savefig(fig_name, dpi=300)
    return


if __name__ == '__main__':
    plot_heatmap('./figures/composition_heatmap.png')
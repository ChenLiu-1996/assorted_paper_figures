import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


results_increase = {
    r'Median property change': [0.1125, 0.44046, 0.46006, 0.45625, 0.46510, 0.4718],
    r'Percentage improved $\uparrow$': [55.15, 70.11, 71.02, 71.46, 71.52, 71.97],
    r'Latent space distance $\downarrow$': [0.11868, 0.11877, 0.11961, 0.11964, 0.11999, 0.11978],
}
results_decrease = {
    r'Median property change': [-1.1246, -1.5761, -1.62873, -1.65457, -1.65872, -1.66086],#, -1.65757],
    r'Percentage improved $\uparrow$': [79.39, 86.31, 87.08, 87.24, 87.01, 86.81],#, 87.32],
    r'Latent space distance $\downarrow$': [0.16017, 0.12758, 0.12141, 0.12024, 0.12015, 0.120326],#, 0.120425],
}

if __name__ == '__main__':
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'helvetica'
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 2

    x_label = 'Optimization step'
    x_values = [1, 5, 10, 20, 40, 80]

    fig = plt.figure(figsize=(15, 4))
    for fig_idx, y_key in enumerate(['Median property change', 'Percentage improved', 'Latent space distance']):
        ax = fig.add_subplot(1, 3, fig_idx + 1)
        for key in results_increase.keys():
            if y_key in key:
                y_values_increase = results_increase[key]
                y_label_increase = key
        for key in results_decrease.keys():
            if y_key in key:
                y_values_decrease = results_decrease[key]
                y_label_decrease = key

        #0F4D92
        ax.plot(x_values,
                y_values_increase,
                linestyle='-', linewidth=3,
                marker='o', markersize=8,
                label='increase property',
                color="#BA9219")
        ax.plot(x_values,
                y_values_decrease,
                linestyle='-', linewidth=3,
                marker='o', markersize=8,
                label='decrease property',
                color='#0F4D92')
        if fig_idx == 1:
            ax.set_ylim([0, 100])
        ax.set_xticks(x_values)
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel(y_label_increase, fontsize=16)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if fig_idx == 2:
            ax.legend(loc='upper right', frameon=False)

    fig.tight_layout(pad=2)
    os.makedirs('./figures', exist_ok=True)
    fig.savefig('./figures/results_sweep.png', dpi=300)

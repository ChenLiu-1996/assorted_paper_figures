import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


summary_label = r'\textit{Improvement}'
options = ['VAE', 'DDPM', 'LDM', 'FM',
           'DiffAb', 'IgLM', 'NOS-C', 'NOS-D',
           'OAE + gradient ascent',
           'OAE + MCMC',
           'OAE + hill climbing',
           'OAE + stochastic hill climbing',
           r'$\texttt{RNAGenScape}$ \textbf{(ours)}',
           summary_label]
colors = ["#cdcdcd", "#767676", "#4d4d4d", "#272727",
          "#c4ece7", "#ecc4c4", "#ecc4e7", "#ea84dd",
          "#d5e29b", "#bdd35c", "#9fbc1d", "#8ead03",
          "#0f4d92"]
xlabel = np.arange(len(options))
xticks = []
results_inference = [0.13, 0.91, 0.74, 5.82, 41.04, 157.57, 0.99, 0.96,
                     0.50, 10.93, 81.52, 99.66, 0.57]
results_openvaccine_delta_pos = [-0.23, -0.33, -0.07, -0.34, 0.06, 0.24, 0.09, 0.18,
                                 -0.01, -0.11, 0.40, -0.12, 0.54]
results_openvaccine_pct_pos = [42.1, 33.8, 47.5, 32.5, 55.0, 63.1, 54.6, 58.3,
                               51.0, 45.1, 69.2, 43.5, 77.5]
results_openvaccine_delta_neg = [-0.23, -0.33, -0.07, -0.34, 0.11, 0.01, -2.25, -0.29,
                                 -0.19, -0.19, -0.29, -0.15, -2.81]
results_openvaccine_pct_neg = [57.9, 66.2, 52.5, 67.5, 44.0, 49.6, 90.6, 65.4,
                               61.3, 57.7, 64.2, 60.0, 97.9]
results_zebrafish_delta_pos = [-0.01, 0.29, -0.95, 0.32, -0.21, -0.57, 0.03, 0.31,
                               -1.21, -0.20, 0.76, 0.32, 0.77]
results_zebrafish_pct_pos = [49.4, 58.2, 22.5, 59.8, 36.3, 32., 51.1, 57.9,
                             18.0, 44.3, 74.4, 60.3, 75.0]
results_zebrafish_delta_neg = [-0.01, 0.29, -0.95, 0.32, -0.21, -0.83, -1.07, 0.38,
                               -0.33, -0.37, -0.87, -0.80, -1.29]
results_zebrafish_pct_neg = [50.6, 41.8, 77.5, 40.2, 60.8, 74.3, 80.8, 40.2,
                             66.5, 60.1, 75.2, 73.7, 85.1]
results_ribosome_delta_pos = [-0.24, -0.11, -0.24, -0.10, -0.05, 0.63, 0.08, -0.09,
                              0.43, -0.19, 0.53, 0.19, 0.63]
results_ribosome_pct_pos = [41.7, 45.9, 41.8, 46.4, 45.0, 80.5, 53.6, 46.5,
                            73.6, 43.0, 83.0, 60.4, 81.4]
results_ribosome_delta_neg = [-0.24, -0.11, -0.24, -0.10, -0.04, -0.51, 0.10, -0.10,
                              -0.05, -0.10, -0.06, -0.05, -0.58]
results_ribosome_pct_neg = [58.3, 54.1, 58.2, 53.6, 54.2, 65.5, 46.0, 53.6,
                            55.7, 54.2, 56.1, 55.2, 67.8]


if __name__ == '__main__':
    PLOT_DE_NOVO_SPEED = False

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'helvetica'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 2

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)

    if PLOT_DE_NOVO_SPEED:
        ax.bar(np.arange(len(options)), 1 / np.array(results_inference), color=colors, label=options)
        ax.hlines(y=1 / results_inference[-1], xmin=-1, xmax=len(options), color=colors[-1], linestyle='--')
        ax.set_xlim([-1, len(options)])
        ax.legend(loc='upper right', frameon=False, fontsize=12)
        ax.set_xticks([])
        ax.set_ylabel('Inference Throughput ' + r'$\uparrow$' +'\n(samples / ms)', fontsize=18)

        # Add braces.
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin - 1, ymax)
        len_de_novo = 4
        mid_de_novo = 0 + (len_de_novo - 0) / 2
        start_opt = len_de_novo + 1
        end_opt = len(options) - 1
        mid_opt = (start_opt + end_opt) / 2
        ax.text(mid_de_novo, ymin - 0.3,
            r'$\underbrace{\rule{5cm}{0pt}}_{\textrm{\textit{de\ novo}\ generative\ models}}$',
            ha='center', va='top', fontsize=14)
        ax.text(mid_opt, ymin - 0.3,
            r'$\underbrace{\rule{9.2cm}{0pt}}_{\textrm{property\ optimization\ methods}}$',
            ha='center', va='top', fontsize=14)
        ax.spines['bottom'].set_position(('data', 0))
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_bounds(0, ymax)

    else:
        # speed_values = 1 / np.array(results_inference[4:])
        # options_speed = options[4:-1]
        # ax.bar(np.arange(len(options_speed)), speed_values, color=colors[4:], label=options_speed)
        speed_values = 1 / np.array(results_inference[4:8] + [results_inference[-1]])
        options_speed = options[4:8] + [options[-2]]
        ax.bar(np.arange(len(options_speed)), speed_values, color=colors[4:8] + [colors[-1]], label=options_speed)
        ax.set_xlim([-1, len(options_speed)])
        ax.legend(loc='upper left', frameon=False, fontsize=16, ncol=1)
        ax.set_xticks([])
        ax.set_ylabel('Inference Throughput ' + r'$\uparrow$' +'\n(samples / ms)', fontsize=18)
        ax.set_yscale('log')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 20)

        # Add numbers on top of bars
        for i, val in enumerate(speed_values):
            ax.text(i, val * 1.1, f'{val:.3f}', ha='center', va='bottom', fontsize=16)

    fig.tight_layout(pad=1)
    os.makedirs('./figures', exist_ok=True)
    fig.savefig('./figures/results_comparison_speed.png', dpi=300)
    plt.close(fig)


    fig = plt.figure(figsize=(20, 9))
    ax = fig.add_subplot(1, 2, 1)
    improvements = np.stack([results_openvaccine_delta_pos, results_openvaccine_delta_neg,
                             results_zebrafish_delta_pos, results_zebrafish_delta_neg,
                             results_ribosome_delta_pos, results_ribosome_delta_neg], axis=1)
    base_improvements = improvements[:-1]
    best_improvements = []
    for j in range(base_improvements.shape[1]):
        col = base_improvements[:, j]
        best_improvements.append(col.max() if j % 2 == 0 else col.min())
    best_improvements = np.array(best_improvements)
    denom = np.where(best_improvements == 0, np.nan, best_improvements)
    improvement_over_best = 100 * (improvements[-1] - best_improvements) / denom
    improvements = np.vstack([improvements, improvement_over_best])
    improvements_display = improvements.copy()
    improvements_display[-1, :] = np.nan
    n_rows, n_cols = improvements.shape
    vmin, vmax = improvements[:-1].min(0), improvements[:-1].max(0)
    cmap_red, cmap_blue = plt.cm.Reds, plt.cm.Blues_r
    for j in range(n_cols):
        cmap = cmap_red if j % 2 == 0 else cmap_blue
        cmap = cmap.copy()
        cmap.set_bad(color="white")
        norm = mpl.colors.Normalize(vmin=vmin[j] if j % 2 == 1 else 0, vmax=vmax[j] if j % 2 == 0 else 0)
        ax.imshow(improvements_display[:, j:j+1], cmap=cmap, norm=norm,
                aspect="auto", extent=[j - 0.5, j + 0.5, 0, n_rows], origin="lower")
    for (i, j), val in np.ndenumerate(improvements):
        if i == n_rows - 1:
            color = "forestgreen" if val >= 0 else "darkred"
            label = f"{val:+.1f} \\%"
            fontsize = 16
        else:
            cmap = cmap_red if j % 2 == 0 else cmap_blue
            norm = mpl.colors.Normalize(vmin=vmin[j] if j % 2 == 1 else 0, vmax=vmax[j] if j % 2 == 0 else 0)
            r, g, b, _ = cmap(norm(val))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            color = "white" if lum < 0.5 else "black"
            label = f"{val:.2f}"
            fontsize = 14
        ax.text(j, i + 0.5, label, ha="center", va="center", fontsize=fontsize, color=color)
    ax.set_title('Median change in property', fontsize=32, pad=24)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([r'$\texttt{OpenVaccine}~(+)$', r'$\texttt{OpenVaccine}~(-)$',
                        r'$\texttt{Zebrafish}~(+)$', r'$\texttt{Zebrafish}~(-)$',
                        r'$\texttt{Ribosome}~(+)$', r'$\texttt{Ribosome}~(-)$'],
                    fontsize=20, rotation=30)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, length=0)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(options, rotation=0, fontsize=20, ha='right')
    for tick in ax.get_yticklabels():
        if r'$\texttt{RNAGenScape}$' in tick.get_text():
            tick.set_fontsize(24)
    ax.set_frame_on(False)
    ax.invert_yaxis()
    # n_rows, n_cols = improvements.shape
    # rect = patches.Rectangle((-0.5, n_rows - 1), n_cols, 1, fill=False, edgecolor='black', linewidth=3)
    # rect.set_clip_on(False)
    # ax.add_patch(rect)

    ax = fig.add_subplot(1, 2, 2)
    percentages = np.stack([results_openvaccine_pct_pos, results_openvaccine_pct_neg,
                            results_zebrafish_pct_pos, results_zebrafish_pct_neg,
                            results_ribosome_pct_pos, results_ribosome_pct_neg], axis=1)
    base_percentages = percentages[:-1]
    best_percentages = base_percentages.max(0)
    pct_improvement_over_best = 100 * (percentages[-1] - best_percentages) / best_percentages
    percentages = np.vstack([percentages, pct_improvement_over_best])
    percentages_display = percentages.copy()
    percentages_display[-1, :] = np.nan
    n_rows, n_cols = percentages.shape
    vmin, vmax = percentages[:-1].min(0), percentages[:-1].max(0)
    cmap_red = plt.cm.Reds.copy()
    cmap_red.set_bad(color="white")
    for j in range(n_cols):
        norm = mpl.colors.Normalize(vmin=max(50, vmin[j]), vmax=vmax[j])
        ax.imshow(percentages_display[:, j:j+1], cmap=cmap_red, norm=norm,
                aspect="auto", extent=[j - 0.5, j + 0.5, 0, n_rows], origin="lower")
    for (i, j), val in np.ndenumerate(percentages):
        if i == n_rows - 1:
            color = "forestgreen" if val >= 0 else "darkred"
            label = f"{val:+.1f} \\%"
            fontsize = 16
        else:
            norm = mpl.colors.Normalize(vmin=max(50, vmin[j]), vmax=vmax[j])
            r, g, b, _ = cmap_red(norm(val))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            color = "white" if lum < 0.5 else "black"
            label = f"{val:.1f} \\%"
            fontsize = 14
        ax.text(j, i + 0.5, label, ha="center", va="center",
                fontsize=fontsize, color=color)
    ax.set_title('Success rate', fontsize=32, pad=24)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([r'$\texttt{OpenVaccine}~(+)$', r'$\texttt{OpenVaccine}~(-)$',
                        r'$\texttt{Zebrafish}~(+)$', r'$\texttt{Zebrafish}~(-)$',
                        r'$\texttt{Ribosome}~(+)$', r'$\texttt{Ribosome}~(-)$'],
                    fontsize=20, rotation=30)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, length=0)
    ax.set_yticks([])
    # ax.set_yticks(np.arange(n_rows) + 0.5)
    # ax.set_yticklabels(options, fontsize=16, ha='right')
    ax.set_frame_on(False)
    ax.invert_yaxis()
    # rect = patches.Rectangle((-0.5, n_rows - 1), n_cols, 1,
    #                         fill=False, edgecolor='black', linewidth=3)
    # rect.set_clip_on(False)
    # ax.add_patch(rect)

    fig.tight_layout(pad=2)
    os.makedirs('./figures', exist_ok=True)
    fig.savefig('./figures/results_comparison_optimization.png', dpi=300)
    plt.close(fig)

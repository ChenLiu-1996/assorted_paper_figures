import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def load_freq_prior_data(data_dir):
    datasets = {}
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".npz"):
            continue
        path = os.path.join(data_dir, name)
        data = np.load(path)
        datasets[name] = {key: data[key] for key in data.files}
    return datasets


def plot_dataset(ax, title, sample1_arr, sample2_arr, max_freq_radius,
                 line_colors, line_styles, legend_labels):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.plot(np.arange(len(sample1_arr))[:max_freq_radius],
            sample1_arr[:max_freq_radius],
            color=line_colors[0], linestyle=line_styles[0], linewidth=2.5)
    ax.plot(np.arange(len(sample2_arr))[:max_freq_radius],
            sample2_arr[:max_freq_radius],
            color=line_colors[1], linestyle=line_styles[1], linewidth=2.5)
    ax.set_title(title, fontfamily='monospace')
    ax.set_xlabel('Frequency Radius')
    ax.set_xticks(np.arange(max_freq_radius)[::4])
    ax.set_ylabel('Mean Amplitude')
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(LinearLocator(4))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.legend(legend_labels, frameon=False, fontsize=14)


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    datasets = load_freq_prior_data(data_dir)

    plt.rcParams['font.family'] = 'helvetica'
    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 0, 1, 1])

    # Part 1. Plot the individual datasets.
    max_freq_radius = 17
    line_colors = ['#2A9D8F', '#E76F51']
    line_styles = ['-', '--']
    subplot_positions = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]

    per_dataset_names = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS']
    max_name_len = max(len(name) for name in per_dataset_names)
    for dataset_idx, dataset_name in enumerate(per_dataset_names):
        data = datasets[dataset_name.lower() + '_freq_prior.npz']
        sample1_arr = data['data1']
        sample2_arr = data['data2']

        ax = fig.add_subplot(subplot_positions[dataset_idx])
        plot_dataset(ax, dataset_name, sample1_arr, sample2_arr,
                     max_freq_radius, line_colors, line_styles,
                     [r'Random subset 1 $(N=50)$', r'Random subset 2 $(N=50)$'])

    # Part 2. Plot the mixed dataset across two columns.
    data = datasets['mixed_freq_prior.npz']
    sample1_arr = data['data1']
    sample2_arr = data['data2']
    ax = fig.add_subplot(gs[0, 3:5])
    plot_dataset(ax, '', sample1_arr, sample2_arr, max_freq_radius, line_colors, line_styles,
                 [r'Random subset 1 $(N=500)$', r'Random subset 2 $(N=500)$'])
    ax.set_title('Mixuture of all four datasets', fontfamily='helvetica')

    # Part 3. Plot per-dataset mean/std curves in a single panel.
    ax = fig.add_subplot(gs[1, 3])
    per_dataset_colors = ['#4C72B0', '#C2A5CF', '#7B3294', '#8C564B']
    curve_means = []
    for dataset_idx, dataset_name in enumerate(['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS']):
        data = datasets[dataset_name.lower() + '_freq_prior.npz']
        sample1_arr = data['data1'][:max_freq_radius]
        sample2_arr = data['data2'][:max_freq_radius]
        stacked = np.stack([sample1_arr, sample2_arr], axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        x_vals = np.arange(len(mean))
        color = per_dataset_colors[dataset_idx]
        padded_name = dataset_name.ljust(max_name_len)
        ax.plot(x_vals, mean, color=color, linewidth=1.5,
                label=f'{padded_name}' + r' $(N=100)$')
        ax.fill_between(x_vals, mean - std, mean + std, color=color, alpha=0.2)
        curve_means.append(mean)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.axvline(2, color='0.6', linestyle='--', linewidth=1.2, label=r'Radius$=2$')
    x_left = ax.get_xlim()[0]
    for mean, color in zip(curve_means, per_dataset_colors):
        ax.hlines(mean[2], xmin=x_left, xmax=2, color=color, linestyle='--', linewidth=1.0)
    ax.set_xlabel('Frequency Radius')
    ax.set_xticks(np.arange(max_freq_radius)[::4])
    ax.set_xlim(left=x_left, right=max_freq_radius)
    ax.set_ylabel('Mean Amplitude')
    ax.set_ylim(bottom=0, top=3)
    ax.yaxis.set_major_locator(LinearLocator(4))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.legend(frameon=False, ncol=1, prop={'family': 'monospace', 'size': 12})

    # Part 4. Plot foreground/background curves in a single panel.
    data = datasets['foreground_background.npz']
    polyp_arr = data['data1'][:max_freq_radius]
    background_arr = data['data2'][:max_freq_radius]
    x_vals = np.arange(len(polyp_arr))
    polyp_color = '#1F3A93'
    background_color = '#16A085'

    ax = fig.add_subplot(gs[1, 4])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    fb_names = ['Polyp', 'Background']
    fb_max_len = max(len(name) for name in fb_names)
    ax.plot(x_vals, polyp_arr, color=polyp_color, linewidth=2.5,
            label=f'{fb_names[0].ljust(fb_max_len)}' + r' $(N=500)$')
    ax.plot(x_vals, background_arr, color=background_color, linewidth=2.5,
            linestyle=':', label=f'{fb_names[1].ljust(fb_max_len)}' + r' $(N=500)$')
    ax.axvline(2, color='0.6', linestyle='--', linewidth=1.2, label=r'Radius$=2$')
    x_left = ax.get_xlim()[0]
    ax.hlines(polyp_arr[2], xmin=x_left, xmax=2, color=polyp_color, linestyle='--', linewidth=1.2)
    ax.hlines(background_arr[2], xmin=x_left, xmax=2, color=background_color, linestyle='--', linewidth=1.2)
    ax.set_xlabel('Frequency Radius')
    ax.set_xticks(np.arange(max_freq_radius)[::4])
    ax.set_xlim(left=x_left, right=max_freq_radius)
    ax.set_ylabel('Mean Amplitude')
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(LinearLocator(4))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.legend(frameon=False, prop={'family': 'monospace', 'size': 12})

    fig.tight_layout(pad=1)
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/freq_prior.png', dpi=300)
import os
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta


DATA = {
    'names': ['Methodological Contribution (Text-only)', 'Evaluation / Application (Text-only)',
              'Methodological Contribution (Multimodal)', 'Evaluation / Application (Multimodal)'],
    'pub_by_month': np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 2, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2],
         [0, 0, 4, 0, 0, 6, 1, 3, 2, 0, 5, 0, 4, 0, 9, 1, 3, 4, 7, 6, 7, 6, 6, 1, 6, 4, 6, 2, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 2, 1, 0, 2, 3, 0, 1, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0]]
    ),
    # NOTE: Use `*` to move the label up in the annotation. Each `*` moves it up a bit.
    'dates_llm': {
        '2022-11': 'ChatGPT\n(GPT-3.5)',
        '2023-02': 'Bard',
        '2023-02': 'LlaMA 1',
        '2023-03': 'GPT-4*',
        '2025-08': 'GPT-5',
        '2023-07': 'LlaMA 2',
        '2024-04': 'LlaMA 3',
        '2025-04': 'LlaMA 4',
        '2023-12': 'Gemini 1.0',
        '2024-02': 'Gemini 1.5',
        '2024-12': 'Gemini 2.0',
        '2025-06': 'Gemini 2.5*',
    },
    'dates_vlm': {
        '2023-02': 'BLIP-2',
        '2023-07': 'LlaVA 1.0',
        '2023-9': 'GPT-4v',
        '2023-10': 'LlaVA 1.5*',
        '2023-12': 'Gemini 1.0',
        '2024-02': 'Gemini 1.5',
        '2024-12': 'Gemini 2.0',
        '2025-06': 'Gemini 2.5',
    }
}

def month_year_list(start_year, start_month, n_months):
    start = datetime(start_year, start_month, 1)
    out = []
    for i in range(n_months):
        d = start + relativedelta(months=i)
        out.append(d.strftime('%Y-%m'))
    return out

def mark_events(ax, time_arr, y_curve, events, dy=0.1):
    x_idx = {t: i for i, t in enumerate(time_arr)}
    y0, y1 = ax.get_ylim()
    prev_date = None
    for date, label in events.items():
        if prev_date is None:
            prev_date = date
        if date in x_idx:
            i = x_idx[date]
            x, y = i, y_curve[i]
            ax.annotate(
                label.replace('*', ''),
                xy=(x, y),
                xytext=(x, y + (1 + 0.8 * np.uint8(label.count('*'))) * dy * (y1 - y0)),
                ha='center',
                va='bottom',
                fontsize=11,
                arrowprops=dict(arrowstyle='-|>', lw=1.3, color='black',
                                shrinkA=0, shrinkB=0, mutation_scale=15)
            )
    return

def plot_curve(fig_name: str):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'helvetica'
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 2
    colors = ["#9BC8FA", "#ffa8a6", "#13457E", "#850c0a"]

    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    fig = plt.figure(figsize=(12, 8))
    num_months = DATA['pub_by_month'].shape[1]

    ax = fig.add_subplot(2, 1, 1)
    time_arr = month_year_list(start_year=2022, start_month=11, n_months=num_months)
    ax.fill_between(time_arr, np.zeros_like(DATA['pub_by_month'][1, :]), np.cumsum(DATA['pub_by_month'][1, :]), color=colors[1],
                    label=DATA['names'][1])
    ax.fill_between(time_arr, np.zeros_like(DATA['pub_by_month'][0, :]), np.cumsum(DATA['pub_by_month'][0, :]), color=colors[0],
                    label=DATA['names'][0])
    ax.plot(time_arr, np.cumsum(DATA['pub_by_month'][0, :]), lw=3, c=colors[2])
    ax.plot(time_arr, np.cumsum(DATA['pub_by_month'][1, :]), lw=3, c=colors[3])
    mark_events(ax, time_arr, np.cumsum(DATA['pub_by_month'][1, :]), DATA['dates_llm'])
    ax.legend(frameon=False)
    ax.set_xticks(time_arr[2::6])
    ax.set_ylabel('Cumulative\nPublication Count\n(Text-only)')

    ax = fig.add_subplot(2, 1, 2)
    time_arr = month_year_list(start_year=2022, start_month=11, n_months=num_months)
    ax.fill_between(time_arr, np.zeros_like(DATA['pub_by_month'][3, :]), np.cumsum(DATA['pub_by_month'][3, :]), color=colors[1],
                    hatch='\\\\\\', edgecolor='black', label=DATA['names'][3])
    ax.fill_between(time_arr, np.zeros_like(DATA['pub_by_month'][3, :]), np.cumsum(DATA['pub_by_month'][3, :]), color=colors[1],
                    facecolor='none', edgecolor='white', linewidth=2) # To visually "erase" the border.
    ax.fill_between(time_arr, np.zeros_like(DATA['pub_by_month'][2, :]), np.cumsum(DATA['pub_by_month'][2, :]), color=colors[0],
                    hatch='///', edgecolor='black', label=DATA['names'][2])
    ax.fill_between(time_arr, np.zeros_like(DATA['pub_by_month'][2, :]), np.cumsum(DATA['pub_by_month'][2, :]), color=colors[0],
                    facecolor='none', edgecolor='white', linewidth=2) # To visually "erase" the border.
    ax.plot(time_arr, np.cumsum(DATA['pub_by_month'][2, :]), lw=3, c=colors[2])
    ax.plot(time_arr, np.cumsum(DATA['pub_by_month'][3, :]), lw=3, c=colors[3])
    ax.legend(frameon=False)
    mark_events(ax, time_arr, np.cumsum(DATA['pub_by_month'][3, :]), DATA['dates_vlm'])
    ax.set_xticks(time_arr[2::6])
    ax.set_ylabel('Cumulative\nPublication Count\n(Multimodal)')

    fig.tight_layout(pad=2)
    fig.savefig(fig_name, dpi=300)
    return


if __name__ == '__main__':
    plot_curve('./figures/trend_by_month.png')
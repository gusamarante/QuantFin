import matplotlib.ticker as plticker
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd


def timeseries(df, title=None, x_major_ticks='year', date_format='%Y', x_label=None, y_label=None,
               fontsize=15, save_path=None, show_chart=False):
    # TODO Documentation

    MyFont = {'fontname': 'Century Gothic'}
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Century Gothic']

    fig = plt.figure(figsize=(12, 12 * 0.61))  # TODO add option to pass the axis
    ax = fig.gca()

    if isinstance(df, pd.Series):
        ax.plot(df.dropna(), label=df.name, color='#0000CD')
    elif isinstance(df, pd.DataFrame):
        for col in df.columns:
            ax.plot(df[col].dropna(), label=col, color='#0000CD')
    else:
        raise ValueError("'df' must be pandas Series or DataFrame")

    ax.tick_params(axis='y', which='both', right=False, left=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=True)

    if x_label is not None:
        plt.xlabel(x_label, MyFont)

    if y_label is not None:
        plt.ylabel(y_label, MyFont)

    ax.set_title(title, fontdict={'fontsize': fontsize + 2, 'fontweight': 'bold'}, **MyFont)

    ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

    if x_major_ticks == 'month':
        locators = mdates.MonthLocator()
        ax.xaxis.set_major_locator(locators)

    elif x_major_ticks == 'year':
        locators = mdates.YearLocator()
        ax.xaxis.set_major_locator(locators)

    plt.xticks(rotation=90)

    x_max, x_min = df.dropna(how='all').index.max(), df.index.dropna(how='all').min()
    ax.set_xlim(x_min - pd.offsets.Day(1), x_max + pd.offsets.Day(1))

    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))

    if (0 > ax.get_ylim()[0]) and (0 < ax.get_ylim()[1]):
        ax.axhline(0, color='black', linewidth=1)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fontsize)

    plt.tight_layout()

    if not (save_path is None):
        plt.savefig(save_path, pad_inches=1, dpi=400)

    if show_chart:
        plt.show()

    plt.close()

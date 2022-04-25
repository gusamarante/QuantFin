from matplotlib.colors import LinearSegmentedColormap
from PyPDF4 import PdfFileReader, PdfFileWriter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
import numpy as np


def timeseries(df, title=None, x_major_ticks='year', date_format='%Y', x_label=None, y_label=None,
               fontsize=15, legend_cols=1, save_path=None, show_chart=False):
    # TODO Documentation

    MyFont = {'fontname': 'Century Gothic'}
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Century Gothic']

    fig = plt.figure(figsize=(12, 12 * 0.61))  # TODO add option to pass the axis
    ax = fig.gca()

    if isinstance(df, pd.Series):
        ax.plot(df.dropna(), label=df.name)
    elif isinstance(df, pd.DataFrame):
        for col in df.columns:
            ax.plot(df[col].dropna(), label=col, linewidth=2)
    else:
        raise ValueError("'df' must be pandas Series or DataFrame")

    ax.tick_params(axis='y', which='both', right=False, left=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=True)

    ax.legend(ncol=legend_cols)

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


def df2pdf(data, col_width=1.2, row_height=0.4, font_size=15,
           header_face_color='#3333B2', header_font_color='w',
           row_colors=('#e6e6e6', 'w'), edge_color='w',
           save_path=None, show_table=False, index_name=None,
           title=None, date_format='%d-%b-%Y', rounding=None):
    """
    :param data: pandas.DataFrame
    :param col_width: width of the columns
    :param row_height: heigth of the rows
    :param font_size: font size for the values. The title's font size will be 'font_size+2'
    :param header_face_color: background color for the row with the column titles
    :param header_font_color: font color for the columns titles
    :param row_colors: tuple with 2 color names or values. The background of the cells
                       will be painted with these colors to facilitate reading.
    :param edge_color: color of the edges of the table
    :param save_path: path to save the table as pdf file
    :param show_table: If True, shows a preview of the table
    :param index_name: Name of the column that will hold the index of the DataFrame
    :param title: Title of the Table
    :param date_format: formatting string to be passed to python's 'strftime', in case
                        the DataFrame has a panda DateTimeIndex
    :param rounding: Number of decimal places to round numeric values.
    :return: matplotlib axis object, in case you want to make further modifications.
    """
    # TODO col_width and row_height should be automatically set based on the length of titles and font_size.

    data = data.copy()

    if rounding is not None:
        data = data.round(rounding)

    if index_name is not None:
        data.index.name = index_name

    if isinstance(data.index, pd.DatetimeIndex):
        data.index = data.index.strftime(date_format)

    data = data.reset_index()

    size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])

    fig, ax = plt.subplots(figsize=size)
    ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=[0, 0, 1, 1], colLabels=data.columns, cellLoc='center')
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table.get_celld().items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color=header_font_color)
            cell.set_facecolor(header_face_color)
        elif k[1] == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    plt.title(title, fontsize=font_size + 2, weight='bold')
    plt.tight_layout()

    if not (save_path is None):
        plt.savefig(save_path)

    if show_table:
        plt.show()

    plt.close()

    return ax.get_figure()


def df2heatmap(data, show_table=False, figsize=(9, 9), nodes=None, colors=None, cmap='rwb', cbar=False,
               fontsize=12, table_title=None, normalize='percentile', save_path=None, ax=None):
    """
    Creates a heatmap table from a pandas dataframe. The heatmap color scale can be based on either the
    percentage of the value relative to the range or based on its percentile.
    @param data: pandas.DataFrame, already sorted in the way you want to see it.
    @param show_table: bool. If True, previews the table during runtime.
    @param figsize: tuple with the dimensions of the figure.
    @param nodes: list of the thresholds (float) to be passed to the matplotlib's colormap building tool.
    @param colors: list of color names or codes (strings) to be passed to the matplotlib's colormap building tool.
    @param cmap: str with name of colormap or matplolib.colormap object.
                 https://matplotlib.org/stable/gallery/color/colormap_reference.html
    @param cbar: bool. If True, plots the colorbar on the figure.
    @param fontsize: Fontsize for the text in the table.
    @param table_title: str, title of the table. Its fontsize is 'fontsize' + 2
    @param normalize: 'percentile' or 'range'. If 'percentile', the heatmap is independent for each column and
                      based on the percentile of the observation. If 'range', the heatmap is independent for
                      each column and based on the percentage of the observation relative to the maximum range.
    @param save_path: str with the path to save the figure. File name must end with '.pdf' or '.png'.
    @param ax: matplotlib.Axis object. Allows to pass an axis that already exists. Used to creat a figure
               with multiple tables
    @return: matplotlib.Figure and matplotlib.Axis objects that can still be manipulated before plotting.
    """
    # TODO heatmap based on the full dataframe and not independent for each column. Add 'percentile-all'
    #  and 'range-all' normalization methods.

    # TODO heatmap based on the values themselves, with the possibilities to choose the range. 'values'

    # TODO Choose to normalize data by Rows or columns

    assert len(colors) == len(nodes), "Length of 'colors' and 'nodes' must be the same"

    myfont = {'fontname': 'Century Gothic'}
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Century Gothic']

    if (nodes is not None) and (colors is not None):
        cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

    # normalize data by columns
    if normalize == 'percentile':
        df_plot = (data.rank() - 1) / (data.shape[0] - 1)

    elif normalize == 'range':
        df_plot = (data - data.min())/(data.max() - data.min())

    else:
        raise AssertionError('Normalization method not implemented.')

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax = sns.heatmap(df_plot, ax=ax, cbar=cbar, annot=data, cmap=cmap,
                     fmt='.2f', annot_kws={'fontsize': fontsize, 'weight': 'normal'},
                     linewidths=1, linecolor='lightgrey')

    ax.xaxis.tick_top()

    ax.set_xticklabels(ax.get_xticklabels(),
                       fontdict={'fontweight': 'bold',
                                 'fontsize': fontsize})

    ax.set_yticklabels(ax.get_yticklabels(),
                       fontdict={'fontweight': 'bold',
                                 'fontsize': fontsize})

    if table_title is not None:
        ax.set_title(table_title,
                     fontdict={'fontweight': 'bold', 'fontsize': fontsize + 2},
                     **myfont)

    plt.tick_params(axis='x', which='both', top=False, bottom=False)
    plt.tick_params(axis='y', which='both', left=False)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fontsize)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    if show_table:
        plt.show()

    return fig, ax


def merge_pdfs(file_path, pdf_writer=None, page=None):
    """
    Utility function to compose a PDF file from parts of other PDF files. The
    output of the function is a PdfFileWriter object from PyPDF4. In order to
    save the resulting object output 'pdf_writer' as a PDF file, use the
    following logic in your routine:

    ```
    with open('save/path/here.pdf', 'wb') as out:
        pdf_writer.write(out)
    ```

    @param file_path: path of the PDF file to get pages from.
    @param pdf_writer: PyPDF4 object PdfFileWriter. If None, starts a new writer.
    @param page: int or list of ints. Number of the page from the file to be added.
                 If None, all pages from 'file_path' are added in order. If list of
                 ints, selected pages are added in the order of the list.
    @return: PdfFileWriter object with the new pages added.
    """

    pdf_reader = PdfFileReader(str(file_path))

    if pdf_writer is None:
        pdf_writer = PdfFileWriter()

    if page is None:
        for pg in range(pdf_reader.getNumPages()):
            pdf_writer.addPage(pdf_reader.getPage(pg))

    else:
        try:
            for pg in page:
                pdf_writer.addPage(pdf_reader.getPage(pg))

        except TypeError:
            pdf_writer.addPage(pdf_reader.getPage(page))

    return pdf_writer

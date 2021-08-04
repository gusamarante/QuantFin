import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

window_days = 21 * 12
ccy1 = 'JPY'
ccy2 = 'BRL'

# file_path = r'/Users/gustavoamarante/Dropbox/CQF/Final Project/'  # Mac
file_path = r'/Users/gusamarante/Dropbox/CQF/Final Project/'  # Macbook

# Read Bloomberg Tickers for renaming
df_tickers = pd.read_excel(file_path + r'Data - BBG Data Values.xlsx',
                           index_col=0, sheet_name='Tickers')

tr_dict = df_tickers['Total Return Index (UBS)'].to_dict()
tr_dict = {v: k for k, v in tr_dict.items()}

# Read Total Return Index
df_tr = pd.read_excel(file_path + r'Data - BBG Data Values.xlsx',
                      index_col=0, sheet_name='Total Return')
df_tr = df_tr.rename(tr_dict, axis=1)

# compute correlations
aux_window = df_tr.rolling(window_days).corr()
aux_window = aux_window.xs(ccy1, level=1, drop_level=True)[ccy2].rename('Rolling Window')

aux_ewm = df_tr.ewm(com=window_days, min_periods=window_days).corr()
aux_ewm = aux_ewm.xs(ccy1, level=1, drop_level=True)[ccy2].rename('Exponentially Weighted')

df_corr = pd.concat([aux_window, aux_ewm], axis=1).dropna()

# Chart
MyFont = {'fontname': 'Century Gothic'}
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Century Gothic']

plt.figure(figsize=(11, 6))
plt.plot(df_corr, linewidth=2)
plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=True)
plt.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=True)
plt.xticks(rotation=45)
plt.legend(df_corr.columns, loc='lower left', ncol=1, frameon=True)
ax = plt.gca()
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.set_xlim((df_tr.index[0], df_tr.index[-1]))
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
# plt.savefig(file_path + r'figures/Correlation change over time.pdf', pad_inches=0)
plt.show()

import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

file_path = r'/Users/gustavoamarante/Dropbox/CQF/Final Project/'  # Mac
# file_path = r'/Users/gusamarante/Dropbox/CQF/Final Project/'  # Macbook

# Read Bloomberg Tickers for renaming
df_tickers = pd.read_excel(file_path + r'Data - BBG Data Values.xlsx',
                           index_col=0, sheet_name='Tickers')

tr_dict = df_tickers['Total Return Index (UBS)'].to_dict()
tr_dict = {v: k for k, v in tr_dict.items()}

# Read Total Return Index
df_tr = pd.read_excel(file_path + r'Data - BBG Data Values.xlsx',
                      index_col=0, sheet_name='Total Return')
df_tr = df_tr.rename(tr_dict, axis=1)

# Compute Momentum
df_mom = df_tr.pct_change(3*21).dropna(how='all')

# Chart
MyFont = {'fontname': 'Century Gothic'}
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Century Gothic']

plt.figure(figsize=(11, 6))
plt.plot(df_mom, linewidth=2)
plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=True)
plt.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=True)
plt.xticks(rotation=45)
plt.legend(df_mom.columns, loc='lower left', ncol=4, frameon=True)
ax = plt.gca()
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.set_xlim((df_mom.index[0], df_mom.index[-1]))
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig(file_path + r'figures/Momentum.pdf', pad_inches=0)
plt.show()

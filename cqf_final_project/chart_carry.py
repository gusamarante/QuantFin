import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

file_path = r'/Users/gustavoamarante/Dropbox/CQF/Final Project/'  # Mac
# file_path = r'/Users/gusamarante/Dropbox/CQF/Final Project/'  # Macbook

# Read Bloomberg Tickers for renaming
df_tickers = pd.read_excel(file_path + r'Data - BBG Data Values.xlsx',
                           index_col=0, sheet_name='Tickers')

fwd_dict = df_tickers['Forward 3m (in bps)'].to_dict()
fwd_dict = {v: k for k, v in fwd_dict.items()}

spot_dict = df_tickers['Spot'].to_dict()
spot_dict = {v: k for k, v in spot_dict.items()}

# Read Forward Price
df_fwd = pd.read_excel(file_path + r'Data - BBG Data Values.xlsx',
                       index_col=0, sheet_name='FWD 3M')
df_fwd = df_fwd.rename(fwd_dict, axis=1)

# Read spot rate
df_spot = pd.read_excel(file_path + r'Data - BBG Data Values.xlsx',
                        index_col=0, sheet_name='Spot')
df_spot = df_spot.rename(spot_dict, axis=1)

# compute the carry
df_carry = (df_spot + df_fwd/10000) / df_spot - 1

# Chart
MyFont = {'fontname': 'Century Gothic'}
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Century Gothic']

df_carry = df_carry.dropna(how='all')

plt.figure(figsize=(11, 6))
plt.plot(df_carry, linewidth=2)
plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=True)
plt.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=True)
plt.xticks(rotation=45)
plt.legend(df_carry.columns, loc='upper left', ncol=2, frameon=True)
ax = plt.gca()
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.set_xlim((df_carry.index[0], df_carry.index[-1]))
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig(file_path + r'figures/Carry.pdf', pad_inches=0)
plt.show()

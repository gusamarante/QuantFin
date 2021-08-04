import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

file_path = r'/Users/gustavoamarante/Dropbox/CQF/Final Project/Data - BBG Data Values.xlsx'

# Read Bloomberg Tickers for renaming
df_tickers = pd.read_excel(file_path, index_col=0, sheet_name='Tickers')

tr_dict = df_tickers['Total Return Index (UBS)'].to_dict()
tr_dict = {v: k for k, v in tr_dict.items()}

df_class = df_tickers['Classification']
df_class = df_class.replace({'DM': '#9FD356',
                             'EM': '#FA824C'})

# Read Total Return Index
df_tr = pd.read_excel(file_path, index_col=0, sheet_name='Total Return')
df_tr = df_tr.rename(tr_dict, axis=1)

# Comnpute returns
df = df_tr.pct_change(1)
corr = df.corr()

# Chart
sns.clustermap(data=corr, method='average', metric='euclidean', figsize=(10, 10), cmap='mako', row_colors=df_class,
               col_colors=df_class, linewidths=0)
plt.savefig('/Users/gustavoamarante/Dropbox/CQF/Final Project/figures/Correlation and Dendrogam.pdf',
            pad_inches=0)
plt.show()

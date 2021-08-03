import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read Bloomberg Tickers for renaming
df_tickers = pd.read_excel(r'/Users/gusamarante/Dropbox/CQF/Final Project/Data - BBG Data Values.xlsx',
                           index_col=0, sheet_name='Tickers')

tr_dict = df_tickers['Total Return Index (UBS)'].to_dict()
tr_dict = {v: k for k, v in tr_dict.items()}

df_class = df_tickers['Classification']
df_class = df_class.replace({'DM': 'green',
                             'EM': 'orange'})

# Read Total Return Index
df_tr = pd.read_excel(r'/Users/gusamarante/Dropbox/CQF/Final Project/Data - BBG Data Values.xlsx',
                      index_col=0, sheet_name='Total Return')
df_tr = df_tr.rename(tr_dict, axis=1)

# Comnpute returns
df = df_tr.pct_change(1)
corr = df.corr()

# Chart
print(corr)
sns.clustermap(data=corr, method='average', metric='euclidean', figsize=(7, 7), cmap='vlag', row_colors=df_class,
               col_colors=df_class, linewidths=1)
plt.show()

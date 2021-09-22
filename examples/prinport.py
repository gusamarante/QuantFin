from quantfin.portfolio import PrincipalPortfolios
import matplotlib.pyplot as plt
import pandas as pd

# file_path = r'C:\Users\gamarante\Dropbox\CQF\Final Project\Data - BBG Data Values.xlsx'  # BW
file_path = r'/Users/gustavoamarante/Dropbox/CQF/Final Project/Data - BBG Data Values.xlsx'  # Mac
# file_path = r'/Users/gusamarante/Dropbox/CQF/Final Project/Data - BBG Data Values.xlsx'  # Macbook


# ===== Read Bloomberg Tickers for renaming =====
df_tickers = pd.read_excel(file_path, index_col=0, sheet_name='Tickers')

tr_dict = df_tickers['Total Return Index (UBS)'].to_dict()
tr_dict = {v: k for k, v in tr_dict.items()}


# ===== Read Total Return Index and compute monthly returns =====
df_tr = pd.read_excel(file_path, index_col=0, sheet_name='Total Return')
df_tr = df_tr.rename(tr_dict, axis=1)

df_ret = df_tr.pct_change(21)
df_ret = df_ret.resample('M').last()
df_ret = df_ret.dropna()


# ===== Compute momentum signal =====
df_mom = df_tr.pct_change(252)
df_mom = df_mom.resample('M').last()
df_mom = df_mom.shift(1)
df_mom = df_mom.dropna()


# ===== Run Principal Portfolios =====
pp = PrincipalPortfolios(df_ret, df_mom)
pp.get_pep(k=1)

from quantfin.charts import timeseries, df2pdf, df2heatmap
from quantfin.portfolio import Performance, Markowitz
from quantfin.data import tracker_feeder, SGS
from quantfin.finmath import compute_eri
import matplotlib.pyplot as plt
from pathlib2 import Path
import pandas as pd
import numpy as np

pd.options.display.max_columns = 50
pd.options.display.width = 250

# Parameters
rf = 0.1265
sharpe_factor = {'BDIV11': 1,
                 'BOVA11': 1,
                 'HASH11': 1,
                 'LTN Curta': 1,
                 'LTN Longa': 1,
                 'NTNB Curta': 1,
                 'NTNB Longa': 1,
                 'NTNF Curta': 1,
                 'NTNF Longa': 1,
                 'SPXI11': 1}
sharpe_factor = pd.Series(data=sharpe_factor, name='Sharpe Factor')

# Grab data
df_tri = tracker_feeder()
df_tri = df_tri.drop(['Cota XP', 'LFT Curta', 'LFT Longa'], axis=1)
df_tri = df_tri[df_tri.index >= '2008-01-01']

# Risk-free
sgs = SGS()
df_cdi = sgs.fetch({12: 'CDI'})
df_cdi = df_cdi['CDI'] / 100

# Compute ERI
df_eri = compute_eri(df_tri, df_cdi)

# Performance Data
perf = Performance(df_eri, skip_dd=True)
print(perf.table)

# generate vols
df_vols = df_eri.pct_change(1).ewm(com=252).std() * np.sqrt(252)
vols = df_vols.iloc[-1]

# generate expected returns
adj_sharpe = (perf.table.loc['Sharpe'] * (1/3) + 0.25 * (1/3)).multiply(sharpe_factor)
exp_ret = adj_sharpe * vols + rf

# generate correlation matrix
df_corr = df_eri.pct_change(1).ewm(com=252).corr()
corr_mat = df_corr.xs('2022-04-14', level=0)

# generate allocation
mkw = Markowitz(mu=exp_ret, sigma=vols, corr=corr_mat, rf=rf, risk_aversion=20, short_sell=False)
print('risky weights', mkw.risky_weights.round(4) * 100)
print('final weights', mkw.weight_p)
mkw.plot()

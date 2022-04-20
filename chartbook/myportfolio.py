from quantfin.portfolio import Performance, Markowitz, SignalWeighted, HRP
from quantfin.charts import timeseries, df2pdf, df2heatmap
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
# print(perf.table)

# ==============================
# ===== Allocation Methods =====
# ==============================
df_weights = pd.DataFrame(columns=df_eri.columns)

# === Markowitz ===
# generate vols
df_vols = df_eri.pct_change(1).ewm(com=252).std() * np.sqrt(252)
vols = df_vols.iloc[-1]

# generate expected returns
adj_sharpe = perf.table.loc['Sharpe'] * (1/3)
exp_ret = adj_sharpe * vols + rf

# generate correlation matrix
df_corr = df_eri.pct_change(1).ewm(com=252).corr()
corr_mat = df_corr.xs('2022-04-14', level=0)

# generate allocation
mkw = Markowitz(mu=exp_ret, sigma=vols, corr=corr_mat, rf=rf, risk_aversion=20, short_sell=False)

df_weights.loc['Markowitz'] = mkw.risky_weights

# === Inverso Vol ===
df_weights.loc['Inverse Vol'] = (1/vols) / (1/vols).sum()

# === Equal Weighted ===
df_weights.loc['Equal'] = 1 / df_eri.shape[1]

# === Hierarchical Risk Parity ===
hrp = HRP(corr_mat)
df_weights.loc['HRP'] = hrp.weights

hrp.plot_dendrogram()
hrp.plot_corr_matrix()

# === End ===
df_weights = df_weights.round(4) * 100
df_weights.loc['Average'] = 100 * df_weights.mean() / df_weights.mean().sum()
df_weights.T.plot.bar()
plt.show()

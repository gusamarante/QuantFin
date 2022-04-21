"""
This routine generates the latest update of my portfolio
"""

from quantfin.portfolio import Performance, Markowitz
from quantfin.data import tracker_feeder, SGS
from quantfin.statistics import corr2cov
from quantfin.finmath import compute_eri
from quantfin.charts import timeseries
import matplotlib.pyplot as plt
from pathlib2 import Path
import seaborn as sns
import pandas as pd
import numpy as np

pd.options.display.max_columns = 50
pd.options.display.width = 250

# Parameters
save_path = Path(r'/Users/gustavoamarante/Dropbox/Personal Portfolio/charts')  # Mac
# save_path = Path(r'C:\Users\gamarante\Dropbox\Personal Portfolio\charts')  # BW
rf = 0.1265
long_run_sharpe = 0.2
chosen_assets = ['LTN Longa', 'NTNF Curta', 'NTNF Longa', 'NTNB Curta', 'NTNB Longa',
                 'BDIV11', 'IVVB', 'BBSD', 'FIND', 'GOVE', 'MATB']

# Grab data
df_tri = tracker_feeder()
df_tri = df_tri[chosen_assets]
df_tri = df_tri[df_tri.index >= '2008-01-01']

# Risk-free
sgs = SGS()
df_cdi = sgs.fetch({12: 'CDI'})
df_cdi = df_cdi['CDI'] / 100

# Compute ERI
df_eri = compute_eri(df_tri, df_cdi)

# Performance Data
perf = Performance(df_eri, skip_dd=True)


# ======================
# ===== Parameters =====
# ======================

# ===== Estimates of Correlation =====
# With Monthly Returns
corr_monthly = df_eri.resample('M').last().pct_change().corr()

# With Daily Returns
last_date = df_eri.index[-1]
df_corr_daily = df_eri.pct_change().ewm(com=252, min_periods=63).corr()
corr_daily = df_corr_daily.xs(last_date)

# Plot all rolling daily correlations
for asset in chosen_assets:
    aux = df_corr_daily.xs(asset, level=1)
    aux = aux.drop(asset, axis=1)
    timeseries(aux.dropna(how='all'), legend_cols=2,
               title=f'EWM Correlations of {asset}',
               save_path=save_path.joinpath(f'{asset} - Correlations.pdf'))

# Clustermap based on daily correlations
sns.clustermap(corr_daily,
               method='single',
               metric='euclidean',
               cmap='vlag')
plt.savefig(save_path.joinpath('Available Assets - Daily Clustermap.pdf'), pad_inches=1, dpi=400)
plt.close()

# Clustermap based on monthly correlations
sns.clustermap(corr_monthly,
               method='single',
               metric='euclidean',
               cmap='vlag')
plt.savefig(save_path.joinpath('Available Assets - Monthly Clustermap.pdf'), pad_inches=1, dpi=400)
plt.close()


# ===== Estimates of Volatility =====
# With Monthly Returns
vols_monthly = df_eri.resample('M').last().pct_change().std() * np.sqrt(12)

# With Daily Returns
last_date = df_eri.index[-1]
df_vols_daily = df_eri.pct_change().ewm(com=252, min_periods=63).std() * np.sqrt(252)
vols_daily = df_vols_daily.loc[last_date]

# Plot volatilities
timeseries(df_vols_daily.dropna(how='all'), legend_cols=2,
           title=f'EWM Volatilities',
           save_path=save_path.joinpath(f'Volatilities.pdf'))

# ===== Expected Returns =====
sharpe_ratios = perf.table.loc['Sharpe'] * (2/3) + long_run_sharpe * (1/3)
expected_returns = sharpe_ratios * vols_daily + rf

# ===== Covariance Matrix =====
cov_mat = corr2cov(corr_daily, vols_daily)

# =======================
# ===== Allocations =====
# =======================

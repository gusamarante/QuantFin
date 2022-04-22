from quantfin.portfolio import Performance, Markowitz, EqualWeights
from quantfin.charts import timeseries, df2pdf
from quantfin.data import tracker_feeder, SGS
from quantfin.statistics import cov2corr
from quantfin.finmath import compute_eri
import matplotlib.pyplot as plt
from pathlib2 import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

pd.options.display.max_columns = 50
pd.options.display.width = 250

# Parameters
show_charts = False
save_path = Path(r'/Users/gustavoamarante/Dropbox/Personal Portfolio')  # Mac
# save_path = Path(r'C:\Users\gamarante\Dropbox\Personal Portfolio')  # BW
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

df_selic = sgs.fetch({432: 'Selic'})
df_rf = (df_selic['Selic'] - 0.1) / 100

# Compute ERI
df_eri = compute_eri(df_tri, df_cdi)

# Performance Data
perf = Performance(df_eri, skip_dd=True, rolling_window=252 * 2)

# =====================
# ===== Backtests =====
# =====================
df_bt = pd.DataFrame()

# ===== Equal Weights =====
bt_ew = EqualWeights(df_eri)
df_bt = pd.concat([df_bt, bt_ew.return_index], axis=1)

# ===== Max Sharpe =====
df_cov = df_eri.pct_change().ewm(com=252, min_periods=63).cov() * 252

rebalance_dates = df_rf.index[df_rf.diff() != 0]
weights_sharpe = pd.DataFrame(columns=df_eri.columns)

for date in tqdm(rebalance_dates, 'Max Sharpe'):

    # check if covariance is available for this date, otherwise continue the loop
    try:
        sigma = df_cov.xs(date)
        sigma = sigma.dropna(how='all', axis=0).dropna(how='all', axis=1)
    except KeyError:
        continue

    # Grab sharpe ratio and check if available
    sharpe = perf.rolling_sharpe.loc[date].dropna()
    if len(sharpe) == 0:
        continue

    available_assets = list(set(sharpe.index).intersection(set(sigma.index)))
    sigma = sigma.loc[available_assets, available_assets]
    sharpe = sharpe.loc[available_assets] * (2/3) + long_run_sharpe * (1/3)

    vols = pd.Series(data=np.sqrt(np.diag(sigma)), index=sharpe.index)
    corr, _ = cov2corr(sigma)
    rf = df_rf.loc[date]
    mu = sharpe * vols + rf

    mkw = Markowitz(mu=mu, sigma=vols, corr=corr, rf=rf, risk_aversion=2, short_sell=False)
    weights_sharpe.loc[date] = mkw.risky_weights

weights_sharpe = weights_sharpe.resample('D').last().fillna(method='ffill')
weights_sharpe = weights_sharpe.reindex(df_eri.index, method='pad')

return_index = (weights_sharpe * df_eri.pct_change(1, fill_method=None)).sum(axis=1, min_count=1).dropna()
return_index = (1 + return_index).cumprod()
return_index = 100 * return_index / return_index.iloc[0]

df_bt = pd.concat([df_bt, return_index.rename('Max Sharpe')], axis=1)

# ===================
# ===== Reports =====
# ===================
perf_bt = Performance(df_bt, skip_dd=False, rolling_window=252 * 2)

timeseries(df_bt, title='Backtests - Excess Return Indexes', show_chart=show_charts,
           save_path=save_path.joinpath('charts/Backtests - Excess Return Index.pdf'))

writer = pd.ExcelWriter(save_path.joinpath(f'Backtests.xlsx'))
perf_bt.table.T.to_excel(writer, 'Backtests')
writer.save()

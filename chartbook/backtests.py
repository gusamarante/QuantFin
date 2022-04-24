from quantfin.portfolio import Performance, EqualWeights, BacktestMaxSharpe, BacktestHRP, HRP
from quantfin.data import tracker_feeder, SGS, DROPBOX
from quantfin.charts import timeseries, df2pdf
from quantfin.statistics import cov2corr
from quantfin.finmath import compute_eri
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

pd.options.display.max_columns = 50
pd.options.display.width = 250

# Parameters
show_charts = False
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
df_rf = df_rf.reindex(df_tri.index)

# Compute ERI
df_eri = compute_eri(df_tri, df_cdi)

# =====================
# ===== Backtests =====
# =====================
df_bt = pd.DataFrame()

# ===== Equal Weights =====
bt_ew = EqualWeights(df_eri)
df_bt = pd.concat([df_bt, bt_ew.return_index], axis=1)
weights_equal = bt_ew.weights.iloc[-1].rename('Equal Weights')

# ===== Max Sharpe =====
rebalance_dates = df_rf.index[df_rf.diff() != 0]

perf = Performance(df_eri, skip_dd=True, rolling_window=252 * 3)
df_cov = df_eri.pct_change().ewm(com=252 * 5, min_periods=63).cov() * 252
df_vols = df_eri.pct_change().ewm(com=252 * 5, min_periods=63).std() * np.sqrt(252)
sharpe = perf.rolling_sharpe.dropna(how='all').fillna(long_run_sharpe) * (2/3) + long_run_sharpe * (1/3)
df_expret = (df_vols * sharpe).add(df_rf, axis=0).dropna(how='all')

bt_ms = BacktestMaxSharpe(eri=df_eri,
                          expected_returns=df_expret,
                          cov=df_cov,
                          risk_free=df_rf,
                          rebalance_dates=rebalance_dates,
                          short_sell=False)

df_bt = pd.concat([df_bt, bt_ms.return_index], axis=1)
weights_maxsharpe = bt_ms.weights.iloc[-1].rename('Max Sharpe')

# ===== Hierarchical Risk Parity =====
df_cov = df_eri.pct_change().ewm(com=252 * 5, min_periods=63).cov() * 252

bt_hrp = BacktestHRP(eri=df_eri, cov=df_cov, rebalance_dates=rebalance_dates,
                     method='complete', metric='braycurtis')
df_bt = pd.concat([df_bt, bt_hrp.return_index], axis=1)

hrp = HRP(cov=df_cov.xs(rebalance_dates[-1]), method='complete', metric='braycurtis')
hrp.plot_dendrogram(show_chart=show_charts,
                    save_path=DROPBOX.joinpath(r'charts/HRP - Dendrogram.pdf'))
hrp.plot_corr_matrix(save_path=DROPBOX.joinpath(r'charts/HRP - Correlation matrix.pdf'),
                     show_chart=show_charts)

weights_hrp = bt_hrp.weights.iloc[-1].rename('HRP')

# ===================
# ===== Reports =====
# ===================
perf_bt = Performance(df_bt, skip_dd=False, rolling_window=252)
print(perf_bt.table)

timeseries(df_bt, title='Backtests - Excess Return Indexes', show_chart=show_charts,
           save_path=DROPBOX.joinpath('charts/Backtests - Excess Return Index.pdf'))

timeseries(perf_bt.rolling_return, title='Backtests - Rolling Returns', show_chart=show_charts,
           save_path=DROPBOX.joinpath('charts/Backtests - Rolling Returns.pdf'))

timeseries(perf_bt.rolling_std, title='Backtests - Rolling Vol', show_chart=show_charts,
           save_path=DROPBOX.joinpath('charts/Backtests - Rolling Vol.pdf'))

timeseries(perf_bt.rolling_sharpe, title='Backtests - Rolling Sharpe', show_chart=show_charts,
           save_path=DROPBOX.joinpath('charts/Backtests - Rolling Sharpe.pdf'))

df_weights = pd.concat([weights_equal, weights_maxsharpe, weights_hrp], axis=1)
df_weights['Average Allocation'] = df_weights.mean(axis=1)

writer = pd.ExcelWriter(DROPBOX.joinpath(f'Backtests.xlsx'))
perf_bt.table.T.to_excel(writer, 'Backtests')
df_weights.to_excel(writer, 'Weights')
writer.save()

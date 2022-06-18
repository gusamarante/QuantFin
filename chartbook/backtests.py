from quantfin.portfolio import Performance, EqualWeights, BacktestMaxSharpe, BacktestHRP, HRP, BacktestERC
from quantfin.data import tracker_feeder, SGS, DROPBOX
from quantfin.simulation import Diffusion
from quantfin.finmath import compute_eri
from quantfin.charts import timeseries
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

pd.options.display.max_columns = 50
pd.options.display.width = 250

# Parameters
show_charts = False
long_run_sharpe = 0.2
y_star = 0.5
chosen_assets = ['NTNB Curta', 'NTNB Longa',
                 'NTNF Curta', 'NTNF Longa',
                 'LTN Curta', 'LTN Longa',
                 'BOVA', 'SMAL', 'CMDB', 'FIND',
                 'BDIV',
                 'IVVB', 'EURP']

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

# Rabalence Dates are after each COPOM decision
rebalance_dates = pd.date_range(start=df_rf.index.min(), end=df_rf.index.max(), freq='M')

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

hrp = HRP(cov=df_cov.xs(df_cov.index.get_level_values(0).max()),
          method='complete', metric='braycurtis')
hrp.plot_dendrogram(show_chart=show_charts,
                    save_path=DROPBOX.joinpath(r'charts/HRP - Dendrogram.pdf'))
hrp.plot_corr_matrix(save_path=DROPBOX.joinpath(r'charts/HRP - Correlation matrix.pdf'),
                     show_chart=show_charts)

weights_hrp = bt_hrp.weights.iloc[-1].rename('HRP')

# ===== Equal Risk Contribution =====
df_cov = df_eri.pct_change().ewm(com=252 * 1, min_periods=63).cov() * 252

bt_erc = BacktestERC(eri=df_eri, cov=df_cov, rebalance_dates=rebalance_dates, name=f'ERC')
df_bt = pd.concat([df_bt, bt_erc.return_index], axis=1)

weights_erc = bt_erc.weights.iloc[-1].rename('ERC')

# ===================
# ===== Reports =====
# ===================
# Correlations of assets
df_corr = df_eri.pct_change().ewm(com=252 * 1, min_periods=63).corr()
for asset in tqdm(df_eri.columns, 'Generating Correlations of assets'):
    timeseries(df_corr.xs(asset, level=1).dropna(how='all').drop(asset, axis=1),
               title=f'{asset} - 1y Rolling Correlations',
               show_chart=show_charts, legend_cols=2,
               save_path=DROPBOX.joinpath(f'charts/{asset} - Rolling Correlations of Assets.pdf'))

# Correlation of allocations
df_corr_bt = df_bt.pct_change().ewm(com=252 * 1, min_periods=63).corr()
for alloc in tqdm(df_bt.columns, 'Generating Correlations of allocations'):
    timeseries(df_corr_bt.xs(alloc, level=1).dropna(how='all').drop(alloc, axis=1),
               title=f'{alloc} - 1y Rolling Correlations',
               show_chart=show_charts, legend_cols=2,
               save_path=DROPBOX.joinpath(f'charts/{alloc} - Rolling Correlations of Allocations.pdf'))

# Performance of the allocations
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

# Expected performance of the current portfolio
df_weights = pd.concat([weights_equal, weights_maxsharpe, weights_hrp, weights_erc], axis=1)
df_weights['Average'] = df_weights.mean(axis=1)

exp_ret = (df_expret.iloc[-1] * df_weights['Average']).sum() * y_star + (1 - y_star) * df_rf.iloc[-1]
exp_vol = y_star * np.sqrt(df_weights['Average'].T
                           @ df_cov.xs(df_cov.index.get_level_values(0).max())
                           @ df_weights['Average'])

df_cota = pd.read_excel(DROPBOX.joinpath('Minha cota XP.xlsx'), index_col=0)

diff = Diffusion(T=1, n=252, k=1, initial_price=df_cota.loc['2022-06-01', 'Portfolio'], process_type='gbm',
                 drift=exp_ret, diffusion=exp_vol)

df_cone = pd.concat([diff.theoretical_mean, diff.ci_upper, diff.ci_lower], axis=1)

# Save to Excel
writer = pd.ExcelWriter(DROPBOX.joinpath(f'Backtests.xlsx'))
perf_bt.table.T.to_excel(writer, 'Backtests')
df_weights.to_excel(writer, 'Weights')
df_cone.to_excel(writer, 'Cone')
writer.save()

"""
This routine builds the optimal FIP (Fundo de investimento em participações) portfolio.
The available assets for this portfolio are:
    - BDIV11: Infra / BTG
    - JURO11: Infra / Sparta
"""
from quantfin.portfolio import Performance, EqualWeights, BacktestHRP, BacktestERC
from quantfin.data import tracker_feeder, SGS, DROPBOX
from quantfin.finmath import compute_eri
import matplotlib.pyplot as plt
import pandas as pd

# Benchmark
sgs = SGS()
df_cdi = sgs.fetch({12: 'CDI'})
df_cdi = df_cdi['CDI'] / 100

# Trackers
df_tri = tracker_feeder()
df_tri = df_tri[['BDIV', 'JURO']]

# Excess Returns
df_eri = compute_eri(df_tri, df_cdi)
rebalance_dates = pd.date_range(df_eri.index[0], df_eri.index[-1], freq='M')
df_cov = df_eri.pct_change().ewm(com=252 * 1, min_periods=63).cov() * 252

# ===== PORTFOLIO CONSTRUCTION =====
# Equal Weighted
bt_ew = EqualWeights(df_eri)

# Hierarchical Risk Parity
bt_hrp = BacktestHRP(eri=df_eri, cov=df_cov, rebalance_dates=rebalance_dates,
                     method='complete', metric='braycurtis')

# Equal Risk Contribution
bt_erc = BacktestERC(eri=df_eri, cov=df_cov, rebalance_dates=rebalance_dates, name=f'ERC')

# ===== COMPARE PERFORMANCE =====
df_compare = pd.concat([bt_ew.return_index.rename('EW'),
                        bt_hrp.return_index.rename('HRP'),
                        bt_erc.return_index.rename('ERC')],
                       axis=1)

perf = Performance(df_compare)
chosen_method = perf.table.loc['Sharpe'].astype(float).idxmax()
print('Chosen method for FIP is', chosen_method)

if chosen_method == 'EW':
    weights = bt_ew.weights.iloc[-1].rename('EW Weights')
    tracker = bt_ew.return_index
elif chosen_method == 'HRP':
    weights = bt_hrp.weights.iloc[-1].rename('HRP Weights')
    tracker = bt_hrp.return_index
elif chosen_method == 'ERC':
    weights = bt_erc.weights.iloc[-1].rename('ERC Weights')
    tracker = bt_hrp.return_index

writer = pd.ExcelWriter(DROPBOX.joinpath(f'Pillar FIP.xlsx'))
weights.to_excel(writer, 'FIP Weights')
tracker.to_excel(writer, 'FIP Trackrers')
writer.save()

"""
This routine builds the optimal FIP (Fundo de investimento em participações) portfolio.
The available assets for this portfolio are:
    - BDIV11: Infra / BTG
    - JURO11: Infra / Sparta
"""
from quantfin.portfolio import Performance, EqualWeights, BacktestHRP
from quantfin.data import tracker_feeder, SGS, DROPBOX
from quantfin.finmath import compute_eri
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

# ===== PORTFOLIO CONSTRUCTION =====
# Equal Weighted
bt_ew = EqualWeights(df_eri)

# Hierarchical Risk Parity
df_cov = df_eri.pct_change().ewm(com=252 * 2, min_periods=63).cov() * 252

rebalance_dates = pd.date_range(df_eri.index[0], df_eri.index[-1], freq='M')
bt_hrp = BacktestHRP(eri=df_eri, cov=df_cov, rebalance_dates=rebalance_dates,
                     method='complete', metric='braycurtis')

# hrp = HRP(cov=df_cov.xs(df_cov.index.get_level_values(0).max()),
#           method='complete', metric='braycurtis')
# hrp.plot_dendrogram(show_chart=show_charts,
#                     save_path=DROPBOX.joinpath(r'charts/HRP - Dendrogram.pdf'))
# hrp.plot_corr_matrix(save_path=DROPBOX.joinpath(r'charts/HRP - Correlation matrix.pdf'),
#                      show_chart=show_charts)

# weights_hrp = bt_hrp.weights.iloc[-1].rename('HRP')

a = 1
# TODO ERC
# TODO HRP

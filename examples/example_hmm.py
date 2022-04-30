from quantfin.data import tracker_feeder, SGS
from quantfin.statistics import GaussianHMM
from quantfin.finmath import compute_eri
import matplotlib.pyplot as plt
import pandas as pd

pd.options.display.max_columns = 50
pd.options.display.width = 250

# Parameters
show_charts = False
chosen_assets = ['NTNB Longa', 'NTNF Longa', 'BOVA']

# SGS
sgs = SGS()
df_sgs = sgs.fetch({12: 'CDI',
                    1: 'BRL'})
df_cdi = df_sgs['CDI'] / 100

# Grab data
df_tri = tracker_feeder()
df_tri = df_tri[chosen_assets]
df_tri = df_tri.dropna(how='all')
df_tri['BRL'] = df_sgs['BRL']

# Compute ERI
df_eri = compute_eri(df_tri, df_cdi)

# Get HMM
hmm = GaussianHMM(returns=df_eri.resample('M').last().pct_change().dropna())
# hmm.select_order(show_chart=True, select_iter=20)
hmm.fit(n_states=3, fit_iter=100)

# attributes
print(hmm.score)
print(hmm.trans_mat.round(3)*100, '\n')
print(hmm.avg_duration.round(1), '\n')
print(hmm.state_freq.round(3)*100)
print(hmm.stationary_dist.round(3)*100, '\n')

# Plots
hmm.digraph()

hmm.predicted_state.plot(title='Predicted State')
plt.tight_layout()
plt.show()

hmm.state_probs.plot(title='State Probabilities')
plt.tight_layout()
plt.show()

hmm.means.plot(kind='bar', title='Means')
plt.tight_layout()
plt.show()

hmm.vols.plot(kind='bar', title='Vols')
plt.tight_layout()
plt.show()

(hmm.means / hmm.vols).plot(kind='bar', title='Sharpe')
plt.tight_layout()
plt.show()

for asset in df_eri.columns:
    hmm.plot(data=df_tri[asset].resample('M').last())

    hmm.corrs.xs(asset, level=1).drop(asset, axis=1).plot(kind='bar', title=f'Correlations of {asset}')
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout()
    plt.show()

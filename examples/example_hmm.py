from quantfin.data import tracker_feeder, SGS
from quantfin.statistics import GaussianHMM
from quantfin.finmath import compute_eri
import matplotlib.pyplot as plt
import pandas as pd

pd.options.display.max_columns = 50
pd.options.display.width = 250

# Parameters
show_charts = False
chosen_assets = ['NTNB Longa', 'NTNF Longa', 'IVVB', 'BOVA']

# Grab data
df_tri = tracker_feeder()
df_tri = df_tri[chosen_assets]
df_tri = df_tri.resample('M').last()
df_tri = df_tri.dropna(how='all')

# Risk-free
sgs = SGS()
df_cdi = sgs.fetch({12: 'CDI'})
df_cdi = df_cdi['CDI'] / 100

# Compute ERI
df_eri = compute_eri(df_tri, df_cdi)

# Get HMM
hmm = GaussianHMM(returns=df_eri.pct_change(1).dropna())
# hmm.select_order(show_chart=True)
hmm.fit(n_states=2, fit_iter=100)

print(hmm.score)
print(hmm.trans_mat.round(3)*100, '\n')
print(hmm.avg_duration.round(1), '\n')
print(hmm.state_freq.round(3)*100)
print(hmm.stationary_dist.round(3)*100, '\n')

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

hmm.corrs.xs('NTNB Longa', level=1).drop('NTNB Longa', axis=1).plot(kind='bar', title='Correlations of NTNB')
plt.tight_layout()
plt.show()

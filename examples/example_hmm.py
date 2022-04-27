from quantfin.data import tracker_feeder, SGS
from quantfin.statistics import GaussianHMM
from quantfin.finmath import compute_eri
import matplotlib.pyplot as plt
import pandas as pd

pd.options.display.max_columns = 50
pd.options.display.width = 250

# Parameters
show_charts = False
chosen_assets = ['NTNF Longa', 'BOVA']

# Grab data
df_tri = tracker_feeder()
df_tri = df_tri[chosen_assets]
df_tri = df_tri.dropna()

# Risk-free
sgs = SGS()
df_cdi = sgs.fetch({12: 'CDI'})
df_cdi = df_cdi['CDI'] / 100

# Compute ERI
df_eri = compute_eri(df_tri, df_cdi)

# Get HMM
hmm = GaussianHMM(df_eri=df_eri, n_states=None, select_iter=10)

# hmm.state_means.plot(kind='bar')
# plt.show()
#
# hmm.state_stds.plot(kind='bar')
# plt.show()
#
# hmm.state_sharpe.plot(kind='bar')
# plt.show()

print(hmm.trans_mat.round(3) * 100)

hmm.state_selection.plot()
plt.show()

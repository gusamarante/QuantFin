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
hmm.fit(n_states=2, fit_iter=20)

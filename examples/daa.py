from quantfin.data import tracker_feeder, SGS
from quantfin.statistics import GaussianHMM
from quantfin.finmath import compute_eri
from quantfin.portfolio import DAACosts, DAALinearCosts
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
hmm.fit(n_states=3, fit_iter=100)

# DAA - Testing with pandas input
Lambda0 = (15 / 10000) * np.eye(df_eri.shape[1])
Lambda1 = (20 / 10000) * np.eye(df_eri.shape[1])
Lambda2 = (50 / 10000) * np.eye(df_eri.shape[1])
Lambda = np.array([Lambda0, Lambda1, Lambda2])

allocations = np.array([100, 100, 100, 100])

print(hmm.trans_mat.round(3)*100)

daa = DAACosts(means=hmm.means,
               covars=hmm.covars,
               costs=Lambda,
               transition_matrix=hmm.trans_mat,
               current_allocation=allocations,
               risk_aversion=1,
               discount_factor=0.99,
               include_returns=True,
               normalize=True)

print('allocations', daa.allocations)
print('aims', daa.aim_portfolios)
print('markowitz', daa.markowitz_portfolios)

ldaa = DAALinearCosts(means=hmm.means,
                      covars=hmm.covars,
                      costs=Lambda,
                      transition_matrix=hmm.trans_mat,
                      current_allocation=allocations,
                      risk_aversion=1,
                      discount_factor=0.99,
                      include_returns=True,
                      normalize=True)

print('linear allocations', ldaa.allocations)
print('linear markowitz', ldaa.markowitz_portfolios)

# TODO ternary plot com a evolução do portfolio

from quantfin.statistics import empirical_covariance, shrink_cov, marchenko_pastur, cov2corr, targeted_shirinkage, \
    ledoitwolf_cov, detone_corr
from quantfin.data import tracker_feeder, SGS
from quantfin.finmath import compute_eri
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Visualization options
pd.options.display.max_columns = 50
pd.options.display.width = 250

# Grab total return indexes
df = tracker_feeder()
df = df[df.index >= '2010-01-01']

# Grab funding series
sgs = SGS()
df_cdi = sgs.fetch({12: 'CDI'})
df_cdi = df_cdi / 100

# Compute ERIs
df_eri = compute_eri(total_return_index=df, funding_return=df_cdi['CDI'])
df_returns = df_eri.pct_change(1).dropna()

# Correlation
emp_cov = empirical_covariance(df_returns)
emp_corr, _ = cov2corr(emp_cov)
# print(emp_corr, '\n')

# Shirinkage
shrunk_cov = shrink_cov(df_returns, alpha=0.5)
shrunk_corr, _ = cov2corr(shrunk_cov)
# print(shrunk_corr, '\n')

# Marchenko-Pastur
mp_cov, _, _ = marchenko_pastur(df_returns)
mp_corr, _ = cov2corr(mp_cov)
# print(mp_corr, '\n')

# Targeted Shrinkage
ts_cov, _, _ = targeted_shirinkage(df_returns, ts_alpha=0.5)  # TODO deveria dar o MP?
ts_corr, _ = cov2corr(ts_cov)
# print(ts_corr, '\n')

# Ledoit-Wolfe
lw_cov = ledoitwolf_cov(df_returns)
lw_corr, _ = cov2corr(lw_cov)
# print(lw_corr, '\n')

# Detoning
detone_corr = detone_corr(emp_corr, n=1)
print(detone_corr, '\n')

# Plot
dif = detone_corr - emp_corr
print(dif)

print(np.abs(dif).max().max())

from quantfin.statistics import empirical_correlation, shrink_cov
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

# Correlation
df_returns = df_eri.pct_change(1)
emp_corr = empirical_correlation(df_returns)
print(emp_corr)

# Shirinkage
shrunk_corr = shrink_cov(emp_corr, alpha=0.5)
print(shrunk_corr)

# Plot

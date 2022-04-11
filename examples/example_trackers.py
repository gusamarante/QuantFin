from quantfin.statistics import empirical_correlation,rescale_vol
from quantfin.data import tracker_feeder, SGS
from quantfin.finmath import compute_eri
import matplotlib.pyplot as plt
import pandas as pd

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

# rescale vol
df_eri_adj = rescale_vol(df_eri)

# Plot
df_eri_adj.plot()
plt.show()

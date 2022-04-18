from quantfin.data import tracker_feeder, SGS
from quantfin.portfolio import Performance, EqualWeights
from quantfin.finmath import compute_eri
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Visualization options
pd.options.display.max_columns = 50
pd.options.display.width = 250

# Grab total return indexes
df = tracker_feeder()
df = df.interpolate(limit_area='inside')
df = df.drop(['Cota XP', 'LFT Curta', 'LFT Longa'], axis=1)
df = df[df.index >= '2007-01-01']

# Grab funding series
sgs = SGS()
df_cdi = sgs.fetch({12: 'CDI'})
df_cdi = df_cdi / 100

# Compute ERIs
df_eri = compute_eri(total_return_index=df, funding_return=df_cdi['CDI'])
df_returns = df_eri.pct_change(1).dropna()

# Equal Weights
ew = EqualWeights(df_eri)
df_eri = pd.concat([df_eri, ew.return_index], axis=1)

# Inverse Vol
# TODO this is a particular case o "signal weighted" with the vol as the signal, so I better work on signal weighted


# Viz
df_eri.plot()
plt.show()

perf = Performance(df_eri)
print(perf.table)

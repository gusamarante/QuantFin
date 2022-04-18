from quantfin.data import tracker_feeder, SGS
from quantfin.portfolio import Performance, EqualWeights, SignalWeighted
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
df_returns = df_eri.pct_change(1)

# Equal Weights
ew = EqualWeights(df_eri, name='Equal')


# Signal Weighted
df_vol = df_returns.rolling(252).std() * np.sqrt(252)
volw = SignalWeighted(trackers=df_eri, signals=df_vol, scheme='value', lag_signals=True, name='Vol')

# Signal Weighted
df_vol = 1 / df_vol
ivolw = SignalWeighted(trackers=df_eri, signals=df_vol, scheme='value', lag_signals=True, name='Inv Vol')


# Ending
df_strats = pd.concat([ew.return_index, volw.return_index, ivolw.return_index], axis=1)

perf = Performance(df_strats)
print(perf.table)

df_strats.plot()
plt.show()

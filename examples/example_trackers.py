from quantfin.portfolio import Performance
from quantfin.data import tracker_feeder
import matplotlib.pyplot as plt
import pandas as pd

pd.options.display.max_columns = 50
pd.options.display.width = 250

df = tracker_feeder()
df = df[df.index >= '2010-01-01']

perf = Performance(df)

for var in perf.table.index:
    aux = perf.table.loc[var].sort_values()
    aux.plot(kind='bar', title=var)
    plt.tight_layout()
    plt.show()

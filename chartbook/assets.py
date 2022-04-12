from quantfin.data import tracker_feeder, SGS
from quantfin.portfolio import Performance
from quantfin.finmath import compute_eri
from quantfin.charts import timeseries
import matplotlib.pyplot as plt
from pathlib2 import Path
import pandas as pd

save_path = Path(r'C:\Users\gamarante\Dropbox\Personal Portfolio\charts')

# Grab data
df_tri = tracker_feeder()
df_tri = df_tri.drop(['Cota XP'], axis=1)
df_tri = df_tri[df_tri.index >= '2007-01-01']
df_tri = 100 * df_tri / df_tri.fillna(method='bfill').iloc[0]

sgs = SGS()
df_cdi = sgs.fetch({12: 'CDI'})
df_cdi = df_cdi['CDI'] / 100

# Compute ERI
df_eri = compute_eri(df_tri, df_cdi)

# Performance Data
# perf_t = Performance(df_tri)

# Charts
for asset in df_eri.columns:

    df_plot = df_tri[asset].dropna()
    timeseries(df_plot, title=f'{asset} - Total Return Index',
               show_chart=True)

    # perf_t.plot_drawdowns(asset, show_chart=True,
    #                       save_path=save_path.joinpath(f'{asset} - Drawdowns.pdf'))
    #
    # perf_t.plot_underwater(asset, show_chart=True,
    #                        save_path=save_path.joinpath(f'{asset} - Underwater.pdf'))

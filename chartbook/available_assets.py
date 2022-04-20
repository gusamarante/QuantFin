from quantfin.charts import timeseries, df2pdf, df2heatmap
from quantfin.data import tracker_feeder, SGS
from quantfin.portfolio import Performance
from quantfin.finmath import compute_eri
import matplotlib.pyplot as plt
from pathlib2 import Path
import pandas as pd

# TODO add rolling measures to chartbook

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 200)

show_charts = False
# save_path = Path(r'/Users/gustavoamarante/Dropbox/Personal Portfolio/charts')  # Mac
save_path = Path(r'C:\Users\gamarante\Dropbox\Personal Portfolio\charts')  # BW

# Grab data
df_tri = tracker_feeder()
df_tri = df_tri.drop(['Cota XP'], axis=1)
# df_tri = df_tri[df_tri.index >= '2010-01-01']
df_tri = 100 * df_tri / df_tri.fillna(method='bfill').iloc[0]
df_tri = df_tri.interpolate(limit_area='inside')

sgs = SGS()
df_cdi = sgs.fetch({12: 'CDI'})
df_cdi = df_cdi['CDI'] / 100

# Compute ERI
df_eri = compute_eri(df_tri, df_cdi)
df_eri = df_eri[df_eri.index >= '2010-01-01']

# Performance Data
perf_t = Performance(df_eri)
perf_t.table.to_clipboard()
print('Copied performance table')

# Charts
for asset in df_eri.columns:

    df_plot = df_tri[asset].dropna()
    timeseries(df_plot, title=f'{asset} - Total Return Index',
               show_chart=show_charts,
               save_path=save_path.joinpath(f'{asset} - Total Return Index.pdf'))

    perf_t.plot_drawdowns(asset, show_chart=show_charts,
                          save_path=save_path.joinpath(f'{asset} - Drawdowns.pdf'))

    perf_t.plot_underwater(asset, show_chart=show_charts,
                           save_path=save_path.joinpath(f'{asset} - Underwater.pdf'))

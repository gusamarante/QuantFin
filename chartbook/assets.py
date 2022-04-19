from quantfin.charts import timeseries, df2pdf, df2heatmap
from quantfin.data import tracker_feeder, SGS
from quantfin.portfolio import Performance
from quantfin.finmath import compute_eri
import matplotlib.pyplot as plt
from pathlib2 import Path
import pandas as pd

# TODO add rolling measures to chartbook

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)

show_charts = False
save_path = Path(r'/Users/gustavoamarante/Dropbox/Personal Portfolio/charts')

# Grab data
df_tri = tracker_feeder()
df_tri = df_tri.drop(['Cota XP'], axis=1)
df_tri = df_tri[df_tri.index >= '2010-01-01']
df_tri = 100 * df_tri / df_tri.fillna(method='bfill').iloc[0]

sgs = SGS()
df_cdi = sgs.fetch({12: 'CDI'})
df_cdi = df_cdi['CDI'] / 100

# Compute ERI
df_eri = compute_eri(df_tri, df_cdi)

# Performance Data
perf_t = Performance(df_eri)
df2pdf(perf_t.table.T.sort_values('Sharpe', ascending=False),
       show_table=show_charts,
       rounding=2,
       save_path=save_path.joinpath(f'Performance Table.pdf'))

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

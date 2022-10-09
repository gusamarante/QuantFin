"""
Generates individual charts and filters the assets
"""
from quantfin.data import tracker_feeder, SGS, DROPBOX
from quantfin.portfolio import Performance
from quantfin.finmath import compute_eri
from quantfin.charts import timeseries
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 200)

show_charts = False

# Grab data
df_tri = tracker_feeder()
# df_tri = df_tri.drop(['Cota XP'], axis=1)
# df_tri = df_tri[df_tri.index >= '2010-01-01']
df_tri = 100 * df_tri / df_tri.fillna(method='bfill').iloc[0]
df_tri = df_tri.interpolate(limit_area='inside')

sgs = SGS()
df_cdi = sgs.fetch({12: 'CDI'})
df_cdi = df_cdi['CDI'] / 100

# Compute ERI
df_tri = df_tri[df_tri.index >= '2006-01-01']
df_eri = compute_eri(df_tri, df_cdi)

# Performance Data
perf_t = Performance(df_eri)
df_perf = perf_t.table.T

writer = pd.ExcelWriter(DROPBOX.joinpath(f'Available Assets.xlsx'))
df_perf.to_excel(writer, 'All Assets')
writer.save()

# Charts
for asset in tqdm(df_eri.columns, 'Generating Charts'):

    df_plot = df_eri[asset].dropna()
    timeseries(df_plot, title=f'{asset} - Excess Return Index',
               show_chart=show_charts,
               save_path=DROPBOX.joinpath(f'charts/{asset} - Excess Return Index.pdf'))

    timeseries(perf_t.rolling_return[asset].dropna(), title=f'{asset} - Rolling Return',
               show_chart=show_charts,
               save_path=DROPBOX.joinpath(f'charts/{asset} - Rolling Return.pdf'))

    timeseries(perf_t.rolling_std[asset].dropna(), title=f'{asset} - Rolling Vol',
               show_chart=show_charts,
               save_path=DROPBOX.joinpath(f'charts/{asset} - Rolling Vol.pdf'))

    timeseries(perf_t.rolling_sharpe[asset].dropna(), title=f'{asset} - Rolling Sharpe',
               show_chart=show_charts,
               save_path=DROPBOX.joinpath(f'charts/{asset} - Rolling Sharpe.pdf'))

    perf_t.plot_drawdowns(asset, show_chart=show_charts,
                          save_path=DROPBOX.joinpath(f'charts/{asset} - Drawdowns.pdf'))

    perf_t.plot_underwater(asset, show_chart=show_charts,
                           save_path=DROPBOX.joinpath(f'charts/{asset} - Underwater.pdf'))

from quantfin.data import tracker_feeder, SGS
from quantfin.portfolio import Performance
from quantfin.finmath import compute_eri
from quantfin.charts import timeseries
import matplotlib.pyplot as plt
from pathlib2 import Path
from tqdm import tqdm
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
df_perf = perf_t.table.T
df_perf = df_perf[df_perf['Sharpe'] >= 0]  # Filter positive sharpe
df_score = (df_perf - df_perf.min(axis=0)) / (df_perf.max(axis=0) - df_perf.min(axis=0))  # normalize indicators
df_score['Kurt'] = 1 - df_score['Kurt']  # Invert
df_score['Start Date'] = 1 - df_score['Start Date']  # Invert
df_score = df_score[df_score['Start Date'] >= 0.1]  # Exclude series that are too short

try:
    df_score['Score'] = (1 * df_score['Sharpe']
                         + 0.5 * df_score['Skew']
                         + 0.2 * df_score['Kurt']
                         + 0.5 * df_score['Max DD']) \
                        / (1 + 0.5 + 0.2 + 0.5)
except KeyError:
    df_score['Score'] = (1 * df_score['Sharpe']
                         + 0.5 * df_score['Skew']
                         + 0.2 * df_score['Kurt']) \
                        / (1 + 0.5 + 0.2)

df_perf['Score'] = df_score['Score']
df_perf = df_perf.dropna()

writer = pd.ExcelWriter(r'C:\Users\gamarante\Dropbox\Personal Portfolio\Available Assets.xlsx')
df_perf.to_excel(writer, 'Filtered')
writer.save()

# Charts
for asset in tqdm(df_eri.columns, 'Generating Charts'):

    df_plot = df_tri[asset].dropna()
    timeseries(df_plot, title=f'{asset} - Total Return Index',
               show_chart=show_charts,
               save_path=save_path.joinpath(f'{asset} - Total Return Index.pdf'))

    perf_t.plot_drawdowns(asset, show_chart=show_charts,
                          save_path=save_path.joinpath(f'{asset} - Drawdowns.pdf'))

    perf_t.plot_underwater(asset, show_chart=show_charts,
                           save_path=save_path.joinpath(f'{asset} - Underwater.pdf'))

from quantfin.data import tracker_feeder, DROPBOX, SGS
from quantfin.portfolio import Performance
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

last_year = 2022
start_date = '2007-01-01'

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)

# =====================
# ===== Read Data =====
# =====================
# Total Return Indexes
df_tri = tracker_feeder()
df_tri = df_tri[['NTNB 0.5y', 'NTNB 1y', 'NTNB 1.5y', 'NTNB 2y', 'NTNB 3y', 'NTNB 4y',
                 'NTNB 5y', 'NTNB 6y', 'NTNB 7y', 'NTNB 8y', 'NTNB 9y', 'NTNB 10y']]

# Real Zero Curve
df_raw_curve = pd.DataFrame()
for year in tqdm(range(2003, last_year + 1), 'Reading Curves'):
    aux = pd.read_csv(DROPBOX.joinpath(f'curves/curva_zero_ntnb_{year}.csv'))
    aux = aux.drop(['Unnamed: 0'], axis=1)
    df_raw_curve = pd.concat([df_raw_curve, aux])

df_raw_curve = df_raw_curve.pivot('reference_date', 'du', 'yield')

df_curve = 1 / ((df_raw_curve + 1) ** (df_raw_curve.columns / 252))  # Discount factors
df_curve = np.log(df_curve)  # ln of the dicount factors
df_curve = df_curve.interpolate(limit_area='inside', axis=1, method='index')  # linear interpolations along the lines
df_curve = np.exp(df_curve)  # back to discounts
df_curve = (1 / df_curve) ** (252 / df_curve.columns) - 1
df_curve = df_curve.drop([0], axis=1)

df_curve = df_curve[df_curve.index >= '2006-04-01']  # Filter the dates
df_curve = df_curve[df_curve.columns[df_curve.columns <= 33*252]]  # Filter columns
df_curve.index = pd.to_datetime(df_curve.index)

# ===========================
# ===== Long-run Sharpe =====
# ===========================
df_returns = df_tri / df_tri.iloc[0]
exponent = (252/np.arange(df_tri.shape[0]))
df_returns = df_returns.pow(exponent, axis=0) - 1

df_vols = df_tri.pct_change(1).expanding().std()*np.sqrt(252)
df_sharpe = df_returns / df_vols

# =======================
# ===== Backtesting =====
# =======================
dates2loop = df_tri.dropna(how='all').index
dates2loop = dates2loop[dates2loop >= start_date]

df_weights = pd.DataFrame()
for date in dates2loop:
    vol = df_vols.loc[date]
    mu = 0.5 * df_sharpe.loc[date] * vol

    # TODO Parei aqui
    corr = df_tri.resample('Q').last().pct_change().corr()

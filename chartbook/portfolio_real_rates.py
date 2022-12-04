from quantfin.data import tracker_feeder, DROPBOX, SGS
from tqdm import tqdm
import pandas as pd
import numpy as np

last_year = 2022

# =====================
# ===== Read Data =====
# =====================
# Total Return Indexes
df_tri = tracker_feeder()
df_tri = df_tri[['NTNB 0.5y', 'NTNB 1.5y', 'NTNB 1y', 'NTNB 2y', 'NTNB 3y', 'NTNB 4y',
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

from quantfin.data import tracker_feeder, DROPBOX, SGS
from tqdm import tqdm
import pandas as pd
import numpy as np

last_year = 2022

# =====================
# ===== Read Data =====
# =====================
# Real Zero Curves
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

# Bonds
ntnb = pd.DataFrame()
for year in tqdm(range(2003, last_year + 1), 'Reading Bonds'):
    aux = pd.read_csv(DROPBOX.joinpath(f'trackers/dados_ntnb {year}.csv'), sep=';')
    ntnb = pd.concat([ntnb, aux])

ntnb['reference date'] = pd.to_datetime(ntnb['reference date'])
ntnb['maturity'] = pd.to_datetime(ntnb['maturity'])
ntnb = ntnb.drop(['Unnamed: 0', 'index'], axis=1)

vna = ntnb.groupby('reference date').max()['vna']

# Trackers
df_tri = tracker_feeder()
df_tri = df_tri[['NTNB Curta', 'NTNB 2y', 'NTNB 5y', 'NTNB 10y', 'NTNB Longa']]

# CDI
sgs = SGS()
df_cdi = sgs.fetch({12: 'CDI'})
df_cdi = df_cdi['CDI'] / 100
df_cdi = (1+df_cdi)**252-1

# ===========================
# ===== Compute Signals =====
# ===========================
# Carry
carry_real = (1 + vna.pct_change(5))**(21/5)-1  # ao mês

carry_nominal_1y = ((1+df_curve[252*1])**(252*1/252)) / (((1+df_curve[252*1-21])**((252*1-21)/252)) * ((1+df_cdi)**(1/12))) - 1
carry_nominal_2y = ((1+df_curve[252*2])**(252*2/252)) / (((1+df_curve[252*2-21])**((252*2-21)/252)) * ((1+df_cdi)**(1/12))) - 1
carry_nominal_5y = ((1+df_curve[252*5])**(252*5/252)) / (((1+df_curve[252*5-21])**((252*5-21)/252)) * ((1+df_cdi)**(1/12))) - 1
carry_nominal_10y = ((1+df_curve[252*10])**(252*10/252)) / (((1+df_curve[252*10-21])**((252*10-21)/252)) * ((1+df_cdi)**(1/12))) - 1

carry_1y = (carry_real + carry_nominal_1y).rename('Carry 1y')
carry_2y = (carry_real + carry_nominal_2y).rename('Carry 2y')
carry_5y = (carry_real + carry_nominal_5y).rename('Carry 5y')
carry_10y = (carry_real + carry_nominal_10y).rename('Carry 10y')

carry = pd.concat([carry_1y, carry_2y, carry_5y, carry_10y], axis=1)

carry = carry[carry.index >= '2008-01-01']

a = 1

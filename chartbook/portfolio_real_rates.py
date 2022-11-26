import pandas as pd
import numpy as np

file_path = '/Users/gustavoamarante/Dropbox/Personal Portfolio/curves/'

df_raw = pd.DataFrame()
for year in range(2003, 2023):
    aux = pd.read_csv(file_path + f'curva_zero_ntnb_{year}.csv')
    aux = aux.drop(['Unnamed: 0'], axis=1)
    df_raw = pd.concat([df_raw, aux])

df_raw = df_raw.pivot('reference_date',  'du', 'yield')

# Interpolate
df_curve = 1 / ((df_raw + 1) ** (df_raw.columns/252))  # Discount factors
df_curve = np.log(df_curve)  # ln of the dicount factors
df_curve = df_curve.interpolate(limit_area='inside', axis=1, method='index')  # linear interpolations along the lines
df_curve = np.exp(df_curve)  # back to discounts
df_curve = (1 / df_curve) ** (252 / df_curve.columns) - 1
df_curve = df_curve.drop([0], axis=1)

df_curve = df_curve[df_curve.index >= '2006-04-01']  # Filter the dates
df_curve = df_curve[df_curve.columns[df_curve.columns <= 33*252]]  # Filter columns

a = 1

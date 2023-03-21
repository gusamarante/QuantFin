import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from quantfin.data import DROPBOX

# User defined parameters
min_sample = 21
start_date = '2007-01-01'

# get data
df_raw = pd.DataFrame()
for year in tqdm(range(2006, 2023 + 1), 'Reading Files'):
    aux = pd.read_csv(DROPBOX.joinpath(f'trackers/dados_di1 {year}.csv'),
                      sep=';')
    df_raw = pd.concat([df_raw, aux], axis=0)

df_raw = df_raw.drop(['Unnamed: 0'], axis=1)
df_raw['reference_date'] = pd.to_datetime(df_raw['reference_date'])
df_raw['maturity_date'] = pd.to_datetime(df_raw['maturity_date'])

# build the curve
df_curve = df_raw.pivot('reference_date', 'du', 'rate')
df_curve = df_curve.interpolate(axis=1, method='cubic')
df_curve = df_curve.dropna(how='any', axis=1)
df_curve.index = pd.to_datetime(df_curve.index)

# Organize DV01
df_dv01 = df_raw.pivot(index='reference_date', columns='du', values='dv01')
df_dv01.index = pd.to_datetime(df_dv01.index)

# ===========================
# ===== Full sample PCA =====
# ===========================
pca = PCA(n_components=3)
pca.fit(df_curve.values)

df_var_full = pd.DataFrame(data=pca.explained_variance_ratio_)
df_loadings_full = pd.DataFrame(data=pca.components_.T,
                                columns=['PC 1', 'PC 2', 'PC 3'],
                                index=df_curve.columns)
df_mean_full = pd.DataFrame(data=pca.mean_, index=df_curve.columns,
                            columns=['Médias'])
df_pca_full = pd.DataFrame(data=pca.transform(df_curve.values),
                           columns=['PC 1', 'PC 2', 'PC 3'],
                           index=df_curve.index)

signal = np.sign(df_loadings_full.iloc[0])
df_loadings_full = df_loadings_full * signal
df_pca_full = df_pca_full * signal

# =========================
# ===== Loop for PCAs =====
# =========================
dates2loop = df_curve.index[df_curve.index >= start_date]

df_var_roll = pd.DataFrame(columns=['PC 1', 'PC 2', 'PC 3'])
df_pca_roll = pd.DataFrame(columns=['PC 1', 'PC 2', 'PC 3'])

for date in tqdm(dates2loop, 'Computing Rolling PCs'):
    aux_curve = df_curve.loc[:date]  # Expanding Window

    if aux_curve.shape[0] < min_sample:
        continue

    pca = PCA(n_components=3)
    pca.fit(aux_curve.values)
    current_pcs = pca.transform(aux_curve.values)

    signal = np.sign(pca.components_.T[0])
    aux_loadings = pca.components_.T * signal
    current_pcs = current_pcs * signal

    df_var_roll.loc[date] = pca.explained_variance_ratio_
    df_pca_roll.loc[date] = current_pcs[-1]

df_pca_full['PC 1'].plot()
df_pca_roll['PC 1'].plot()
plt.show()

df_pca_full['PC 2'].plot()
df_pca_roll['PC 2'].plot()
plt.show()

df_pca_full['PC 3'].plot()
df_pca_roll['PC 3'].plot()
plt.show()

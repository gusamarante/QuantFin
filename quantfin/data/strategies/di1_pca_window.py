import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pandas.tseries.offsets import BDay

# User defined parameters
window_size = 252 * 2
min_sample = 21
start_date = '2007-01-01'
exposition_pcadv01 = [0, 0, 1000]
exposition_pca_number = 3
holding_period = 30

# get data
df_raw = pd.read_csv(r'/Users/gustavoamarante/Dropbox/Aulas/Insper - Renda Fixa/2022/Dados DI1.csv', index_col=0)
df_raw['reference_date'] = pd.to_datetime(df_raw['reference_date'])

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
df_loadings_full = pd.DataFrame(data=pca.components_.T, columns=['PC 1', 'PC 2', 'PC 3'], index=df_curve.columns)
df_mean_full = pd.DataFrame(data=pca.mean_, index=df_curve.columns, columns=['Médias'])
df_pca_full = pd.DataFrame(data=pca.transform(df_curve.values), columns=['PC 1', 'PC 2', 'PC 3'], index=df_curve.index)

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
    aux_curve = df_curve.loc[date - BDay(window_size):date]  # Rolling Window
    # aux_curve = df_curve.loc[:date]  # Expanding Window

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

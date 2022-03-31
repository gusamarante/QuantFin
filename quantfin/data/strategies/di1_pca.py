import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pandas.tseries.offsets import BDay

# User defined parameters
window_size = 252 * 2
min_sample = 21
start_date = '2020-01-01'
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


# ====================
# ===== Backtest =====
# ====================
# TODO acompanhar DUs, acompanhar contratos, acompanhar escolhas de DUs, rolling historical percentile, entradas e saídas e neutralidades
# Function that selects the portfolio for a given date
def get_portfolio(current_date, pcadv01, window):
    current_dt = pd.to_datetime(current_date)

    pca = PCA(n_components=3)
    pca.fit(df_curve.loc[current_dt - BDay(window):current_dt].values)

    current_loadings = pd.DataFrame(data=pca.components_, index=range(1, 4), columns=df_curve.columns).T
    current_pca = pd.DataFrame(data=pca.transform(df_curve.values), columns=range(1, 4), index=df_curve.index)

    signal = np.sign(current_loadings.iloc[0])
    current_loadings = current_loadings * signal
    current_pca = current_pca * signal

    available_maturities = df_dv01.loc[current_date].dropna().index
    available_maturities = available_maturities[df_curve.columns.min() <= available_maturities]
    available_maturities = available_maturities[df_curve.columns.max() >= available_maturities]

    aux_pcadv01 = current_loadings.loc[available_maturities].multiply(df_dv01.loc[current_date, available_maturities],
                                                                      axis=0)
    vertices_du = [aux_pcadv01.index[2], aux_pcadv01.idxmax()[3], aux_pcadv01.idxmin()[3]]

    selected_portfolio = pd.DataFrame(index=vertices_du)

    cond_date = df_raw['reference_date'] == current_date
    cond_du = df_raw['du'].isin(vertices_du)
    current_data = df_raw[cond_date & cond_du].sort_values('du')

    selected_portfolio['contracts'] = current_data['contract'].values
    selected_portfolio['pu'] = current_data['theoretical_price'].values
    selected_portfolio[['Loadings 1', 'Loadings 2', 'Loadings 3']] = current_loadings.loc[vertices_du]
    selected_portfolio[['PCADV01 1', 'PCADV01 2', 'PCADV01 3']] = aux_pcadv01.loc[vertices_du]

    selected_signal = pd.Series(index=['PCA 1', 'PCA 2', 'PCA 3'], data=current_pca.loc[current_date].values)

    coeff = selected_portfolio[['PCADV01 1', 'PCADV01 2', 'PCADV01 3']].T.values
    constants = np.array(pcadv01)
    selected_portfolio['quantities'] = np.linalg.inv(coeff) @ constants

    return selected_portfolio, selected_signal


# find the first portfolio
dates2loop = df_curve.index[df_curve.index >= start_date]
df_backtest = pd.DataFrame()

current_portfolios, current_signal = get_portfolio(dates2loop[0], exposition_pcadv01, window_size)
df_backtest.loc[dates2loop[0], 'notional'] = (current_portfolios['quantities'] * current_portfolios['pu']).abs().sum()
df_backtest.loc[dates2loop[0], 'signal'] = current_signal.iloc[exposition_pca_number - 1]
next_roll_date = pd.to_datetime(dates2loop[0]) + pd.offsets.Day(holding_period)

for date, datem1 in tqdm(zip(dates2loop[1:], dates2loop[:-1]), 'Backtesting'):
    filter_contracts = df_raw['contract'].isin(current_portfolios['contracts'].values)
    filter_date = df_raw['reference_date'] == date
    pnl_contracts = df_raw[filter_date & filter_contracts].sort_values('du')['pnl'].values
    pnl_today = current_portfolios['quantities'].values.dot(pnl_contracts)
    df_backtest.loc[date, 'notional'] = df_backtest.loc[datem1, 'notional'] + pnl_today

    if pd.to_datetime(date) >= next_roll_date:
        current_portfolios, current_signal = get_portfolio(datem1, exposition_pcadv01, window_size)
        df_backtest.loc[date, 'signal'] = current_signal.iloc[exposition_pca_number - 1]
        next_roll_date = pd.to_datetime(date) + pd.offsets.Day(holding_period)

df_backtest.index = pd.to_datetime(df_backtest.index)
df_backtest['signal'] = df_backtest['signal'].fillna(method='ffill')
df_backtest['notional'] = 100 * df_backtest['notional'] / df_backtest['notional'].iloc[0]

ax1 = df_backtest['notional'].plot()
ax2 = ax1.twinx()
ax2.spines['right'].set_position(('axes', 1.0))
df_backtest['signal'].plot(ax=ax2, color='orange')
plt.show()
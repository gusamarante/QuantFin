import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

start_date = '2010-01-01'
exposition_pcadv01 = [0, 1000, 0]
exposition_pca_number = 2
holding_period = 30

# read the CSV file - change the file path to work with your computer
df_raw = pd.read_csv(r'/Users/gustavoamarante/Dropbox/Aulas/Insper - Renda Fixa/2022/Dados DI1.csv', index_col=0)

# organize the data
df_curve = df_raw.pivot(index='reference_date', columns='du', values='rate')
df_curve = df_curve.interpolate(method='cubic', axis=1)
df_curve = df_curve.dropna(axis=1)

df_dv01 = df_raw.pivot(index='reference_date', columns='du', values='dv01')

# ==========================================
# ===== PCA analysis for a single date =====
# ==========================================
pca = PCA(n_components=3)
pca.fit(df_curve.values)

df_var = pd.DataFrame(data=pca.explained_variance_ratio_)
df_mean = pd.DataFrame(data=pca.mean_, index=df_curve.columns, columns=['Médias'])
df_loadings = pd.DataFrame(data=pca.components_, index=range(1, 4), columns=df_curve.columns).T
df_pca = pd.DataFrame(data=pca.transform(df_curve.values), columns=range(1, 4), index=df_curve.index)

signal = np.sign(df_loadings.iloc[0])
df_loadings = df_loadings * signal
df_pca = df_pca * signal


# ====================
# ===== Backtest =====
# ====================
# Function that selects the portfolio for a given date
def get_portfolio(current_date, pcadv01):
    df_curve.index = pd.to_datetime(df_curve.index)
    current_dt = pd.to_datetime(current_date)

    pca = PCA(n_components=3)
    pca.fit(df_curve.loc[current_dt - pd.offsets.Day(365*4):current_dt].values)

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
    vertices_du = [aux_pcadv01.index[2], aux_pcadv01.idxmax()[3], aux_pcadv01.index[-1]]

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

current_portfolios, current_signal = get_portfolio(dates2loop[0], exposition_pcadv01)
df_backtest.loc[dates2loop[0], 'notional'] = (current_portfolios['quantities'] * current_portfolios['pu']).abs().sum()
df_backtest.loc[dates2loop[0], 'signal'] = current_signal.iloc[exposition_pca_number - 1]
next_roll_date = pd.to_datetime(dates2loop[0]) + pd.offsets.Day(holding_period)

for date, datem1 in tqdm(zip(dates2loop[1:], dates2loop[:-1])):
    filter_contracts = df_raw['contract'].isin(current_portfolios['contracts'].values)
    filter_date = df_raw['reference_date'] == date
    pnl_contracts = df_raw[filter_date & filter_contracts].sort_values('du')['pnl'].values
    pnl_today = current_portfolios['quantities'].values.dot(pnl_contracts)
    df_backtest.loc[date, 'notional'] = df_backtest.loc[datem1, 'notional'] + pnl_today

    if pd.to_datetime(date) >= next_roll_date:
        current_portfolios, current_signal = get_portfolio(datem1, exposition_pcadv01)
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

"""
Backtest the signal generated from 'di1_pca_full_expanding'. Includes an entry level and an exit level based on the PC
levels.
"""

import getpass
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from quantfin.data import DROPBOX
from scipy.stats import percentileofscore

pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 250)

start_date = '2007-01-01'
min_sample = 21
exposition_pca_number = 3
exposition_pcadv01 = [0, 0, 100]

entry_up = 90
exit_up = 70
entry_down = 100 - entry_up
exit_down = 100 - exit_up

username = getpass.getuser()
save_path = Path(f'/Users/{username}/Dropbox/Aulas/Insper - Renda Fixa/2023')

# Read the DIs
df_di = pd.DataFrame()
for year in tqdm(range(2006, 2023 + 1), 'Reading Files'):
    aux = pd.read_csv(DROPBOX.joinpath(f'trackers/dados_di1 {year}.csv'),
                      sep=';')
    df_di = pd.concat([df_di, aux], axis=0)
df_di = df_di.drop(['Unnamed: 0'], axis=1)
df_di['reference_date'] = pd.to_datetime(df_di['reference_date'])
df_di['maturity_date'] = pd.to_datetime(df_di['maturity_date'])

# build the curve
df_curve = df_di.pivot('reference_date', 'du', 'rate')
df_curve = df_curve.interpolate(axis=1, method='cubic')
df_curve = df_curve.dropna(how='any', axis=1)
df_curve.index = pd.to_datetime(df_curve.index)

# Organize DV01
df_dv01 = df_di.pivot(index='reference_date', columns='du', values='dv01')
df_dv01.index = pd.to_datetime(df_dv01.index)

# Read the signals
df_signal = pd.read_csv(save_path.joinpath('DI1 PCA/expanding_pcs2.csv'),
                        sep=';')
df_signal = df_signal.rename({'Unnamed: 0': 'date'}, axis=1)
df_signal['date'] = pd.to_datetime(df_signal['date'])
df_signal = df_signal.set_index('date')
df_signal = df_signal[['PC 1', 'PC 2', 'PC 3']]

# Read the factor loadings
df_loadings = pd.read_csv(save_path.joinpath('DI1 PCA/expanding_loadings2.csv'),
                          sep=';')
df_loadings = df_loadings.drop(['Unnamed: 0'], axis=1)
df_loadings['date'] = pd.to_datetime(df_loadings['date'])
df_loadings = df_loadings.set_index(['date', 'du'])
df_loadings = df_loadings[['PC 1', 'PC 2', 'PC 3']]


# ===== Custom Function =====
def get_portfolio(current_date, pcadv01):
    """
    given a date and the desired exposition vector, returns the chosen contracts
    """
    current_date = pd.to_datetime(current_date)

    current_loadings = df_loadings.loc[current_date]
    current_pca = df_signal.loc[current_date]

    available_maturities = df_dv01.loc[current_date].dropna().index
    available_maturities = available_maturities[df_curve.columns.min() <= available_maturities]
    available_maturities = available_maturities[df_curve.columns.max() >= available_maturities]

    # TODO add liquidity restriction here

    aux_pcadv01 = current_loadings.loc[available_maturities]
    aux_pcadv01 = aux_pcadv01.multiply(df_dv01.loc[current_date, available_maturities], axis=0)
    vertices_du = [aux_pcadv01.index[5], aux_pcadv01.idxmax().iloc[2], aux_pcadv01.index[-1]]

    selected_portfolio = pd.DataFrame(index=vertices_du)
    cond_date = df_di['reference_date'] == current_date
    cond_du = df_di['du'].isin(vertices_du)
    current_data = df_di[cond_date & cond_du].sort_values('du')

    selected_portfolio['contracts'] = current_data['contract'].values
    selected_portfolio['pu'] = current_data['theoretical_price'].values
    selected_portfolio[['Loadings 1', 'Loadings 2', 'Loadings 3']] = current_loadings.loc[vertices_du]
    selected_portfolio[['PCADV01 1', 'PCADV01 2', 'PCADV01 3']] = aux_pcadv01.loc[vertices_du]
    selected_portfolio[['PC 1', 'PC 2', 'PC 3']] = current_pca.values

    coeff = selected_portfolio[['PCADV01 1', 'PCADV01 2', 'PCADV01 3']].T.values
    constants = np.array(pcadv01)
    selected_portfolio['quantities'] = np.linalg.inv(coeff) @ constants

    return selected_portfolio


# ==== Backtest =====
df_backtest = pd.DataFrame()
dates2loop = df_signal.index[df_signal.index >= start_date]

current_portfolio = get_portfolio(dates2loop[0], exposition_pcadv01)

df_backtest.loc[dates2loop[0], 'notional'] = (current_portfolio['quantities'].abs() * current_portfolio['pu']).sum()
df_backtest.loc[dates2loop[0], 'position'] = 0  # Start with no position

dates2loop = list(zip(dates2loop[1:], dates2loop[:-1]))
for date, datem1 in tqdm(dates2loop, 'Backtesting'):

    # portfolio will be computed everyday
    current_portfolio = get_portfolio(datem1, exposition_pcadv01)  # Signal has a 1 day lag
    df_backtest.loc[date, 'signal'] = current_portfolio[f'PC {exposition_pca_number}'].iloc[0]

    if df_backtest.shape[0] < min_sample:  # assures min_sample observations before rebalances
        df_backtest.loc[date, 'position'] = 0  # stay neutral
        df_backtest.loc[date, 'notional'] = df_backtest.loc[datem1, 'notional']
        continue

    # Relative position of the signal
    signal_position = percentileofscore(a=df_backtest['signal'].dropna(),
                                        score=df_backtest.loc[date, 'signal'],
                                        kind='weak')

    # signal_position = 100*(df_backtest.loc[date, 'signal'] - df_backtest['signal'].min())/(df_backtest['signal'].max() - df_backtest['signal'].min())

    df_backtest.loc[date, 'signal position'] = signal_position

    # Define position
    if df_backtest.loc[datem1, 'position'] == 1:
        # If yesterday I had a long position, check if I can exit
        signm1 = np.sign(df_backtest.loc[datem1, 'signal position'] - exit_down)
        sign = np.sign(df_backtest.loc[date, 'signal position'] - exit_down)

        if sign == signm1:
            # Still on long levels, hold the long position
            df_backtest.loc[date, 'position'] = 1
        else:
            # Crossed the threshold. Exit the position
            df_backtest.loc[date, 'position'] = 0

    elif df_backtest.loc[datem1, 'position'] == -1:
        # If yesterday I had a short position, check if I can exit
        signm1 = np.sign(df_backtest.loc[datem1, 'signal position'] - exit_up)
        sign = np.sign(df_backtest.loc[date, 'signal position'] - exit_up)

        if sign == signm1:
            # Still on short levels, hold the short position
            df_backtest.loc[date, 'position'] = -1
        else:
            # Crossed the threshold. Exit the position
            df_backtest.loc[date, 'position'] = 0
    else:
        # If yesterday I had no position, check if I should enter one

        if df_backtest.loc[date, 'signal position'] >= entry_up:
            # Build a short position, invert the quantities
            df_backtest.loc[date, 'position'] = -1
        elif df_backtest.loc[date, 'signal position'] <= entry_down:
            # Build a long position
            df_backtest.loc[date, 'position'] = 1
        else:
            # stay out
            df_backtest.loc[date, 'position'] = 0

    # Today's PnL
    if np.abs(df_backtest.loc[date, 'position']) == 1:
        # If I am positioned
        filter_contracts = df_di['contract'].isin(current_portfolio['contracts'].values)
        filter_date = df_di['reference_date'] == date
        pnl_contracts = df_di[filter_date & filter_contracts].sort_values('du')['pnl'].values
        pnl_today = (df_backtest.loc[date, 'position'] * current_portfolio['quantities']).values.dot(pnl_contracts)
        df_backtest.loc[date, ['du 1', 'du 2', 'du 3']] = current_portfolio.index.sort_values()
        df_backtest.loc[date, ['q 1', 'q 2', 'q 3']] = current_portfolio['quantities'].values

    else:
        # If I am not positioned
        pnl_today = 0
        df_backtest.loc[date, ['du 1', 'du 2', 'du 3']] = [0, 0, 0]
        df_backtest.loc[date, ['q 1', 'q 2', 'q 3']] = [0, 0, 0]

    df_backtest.loc[date, 'pnl'] = pnl_today
    df_backtest.loc[date, 'notional'] = df_backtest.loc[datem1, 'notional'] + pnl_today

df_backtest.to_clipboard()
df_backtest['notional'].plot()
plt.show()

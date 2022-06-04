import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

notional = 100
holding_period = 21
transaction_cost = 0

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)

df_raw = pd.read_csv(r'/Users/gustavoamarante/Dropbox/Personal Portfolio/trackers/Dados DI1.csv',
                     sep=';', index_col=0)

df_raw['reference_date'] = pd.to_datetime(df_raw['reference_date'])
df_raw['maturity_date'] = pd.to_datetime(df_raw['maturity_date'])

df_raw.loc[119290, 'pnl'] = -76.86 * (254 / 296)  # Correcting a wrong value

time_menu = pd.to_datetime(sorted(df_raw['reference_date'].unique()))

# ===== BACKTEST =====
df_trackers = pd.DataFrame()

for avg_mat in np.arange(0.5, 10.5, 0.5):

    df_backtest = pd.DataFrame()
    start_du = int((2 * avg_mat * 252 + holding_period) / 2)  # This assures that, on average, the position maturity is avg_du

    # first day - build initial position
    first_date = df_raw[df_raw['du'] >= start_du]['reference_date'].min()

    aux = df_raw[df_raw['reference_date'] == first_date].sort_values('du')
    pos = np.searchsorted(aux['du'].values, start_du, side='right')
    current_contracts = aux.iloc[[pos - 1, pos]]
    A = np.vstack([current_contracts['du'].values, np.ones((2,))])
    B = np.array([[start_du], [1]])
    shares = np.linalg.inv(A) @ B

    df_backtest.loc[first_date, 'contract 1'] = current_contracts.iloc[0]['contract']
    df_backtest.loc[first_date, 'contract 2'] = current_contracts.iloc[1]['contract']
    df_backtest.loc[first_date, 'quant 1'] = notional * shares[0, 0] / current_contracts.iloc[0]['theoretical_price']
    df_backtest.loc[first_date, 'quant 2'] = notional * shares[1, 0] / current_contracts.iloc[1]['theoretical_price']
    df_backtest.loc[first_date, 'notional'] = notional

    next_rebalance = first_date + pd.offsets.BDay(holding_period)

    dates2loop = time_menu[time_menu >= first_date].copy()

    for date, datem1 in tqdm(zip(dates2loop[1:], dates2loop[:-1]), f'{avg_mat} years'):

        if date >= next_rebalance:
            aux = df_raw[df_raw['reference_date'] == date].sort_values('du')
            pos = np.searchsorted(aux['du'].values, start_du, side='right')

            try:
                current_contracts = aux.iloc[[pos - 1, pos]]
                A = np.vstack([aux['du'].values[[pos - 1, pos]], np.ones((2,))])
            except IndexError:
                current_contracts = aux.iloc[-2:]
                A = np.vstack([aux['du'].values[-2:], np.ones((2,))])

            B = np.array([[start_du], [1]])
            shares = np.linalg.inv(A) @ B

            df_backtest.loc[date, 'contract 1'] = current_contracts.iloc[0]['contract']
            df_backtest.loc[date, 'contract 2'] = current_contracts.iloc[1]['contract']
            df_backtest.loc[date, 'quant 1'] = df_backtest.loc[datem1, 'notional'] * shares[0, 0] / current_contracts.iloc[0]['theoretical_price']
            df_backtest.loc[date, 'quant 2'] = df_backtest.loc[datem1, 'notional'] * shares[1, 0] / current_contracts.iloc[1]['theoretical_price']
            cost = (df_backtest.loc[date, 'quant 1'] + df_backtest.loc[date, 'quant 2'] + df_backtest.loc[datem1, 'quant 1'] + df_backtest.loc[datem1, 'quant 2']) * transaction_cost

            next_rebalance = date + pd.offsets.BDay(holding_period)

        else:
            df_backtest.loc[date, 'quant 1'] = df_backtest.loc[datem1, 'quant 1']
            df_backtest.loc[date, 'quant 2'] = df_backtest.loc[datem1, 'quant 2']
            df_backtest.loc[date, 'contract 1'] = df_backtest.loc[datem1, 'contract 1']
            df_backtest.loc[date, 'contract 2'] = df_backtest.loc[datem1, 'contract 2']
            cost = 0

        cond_date = df_raw['reference_date'] == date
        cond_contract = df_raw['contract'].isin(df_backtest.loc[date, ['contract 1', 'contract 2']].values)

        aux = df_raw[cond_date & cond_contract]

        df_backtest.loc[date, 'notional'] = df_backtest.loc[datem1, 'notional'] \
                                            + aux.iloc[0]['pnl'] * df_backtest.loc[date, 'quant 1'] \
                                            + aux.iloc[1]['pnl'] * df_backtest.loc[date, 'quant 2'] \
                                            - cost

    df_trackers = pd.concat([df_trackers, df_backtest['notional'].rename(f'{avg_mat} years')], axis=1)

df_trackers.to_csv(r'/Users/gustavoamarante/Dropbox/Personal Portfolio/trackers/trackers_DI1.csv',
                   sep=';')

df_trackers.plot(title=f'DI Trackers')
plt.show()

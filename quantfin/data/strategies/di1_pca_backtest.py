import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import percentileofscore
from pandas.tseries.offsets import BDay, Day

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# User defined parameters
window_size = 252 * 2  # TODO Hyper
min_sample = 21  # TODO Hyper
start_date = '2007-01-01'
exposition_pcadv01 = [0, 0, 1000]
exposition_pca_number = 3
holding_period = 30  # TODO Hyper
entry_bound = 10  # Must be below 50 # TODO Hyper

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


# ====================
# ===== Backtest =====
# ====================
# TODO rolling historical percentile, entradas e saídas e neutralidades
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

    if len(set(vertices_du)) != 3:
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


# Begin
df_backtest = pd.DataFrame()
dates2loop = df_curve.index[df_curve.index >= start_date]
df_backtest.loc[dates2loop[0], 'position'] = 0

dates2loop = zip(dates2loop[1:], dates2loop[:-1])
for date, datem1 in tqdm(dates2loop, 'Backtesting'):
    # Signal and portfolio will be computed everyday
    current_portfolios, current_signal = get_portfolio(datem1, exposition_pcadv01, window_size)
    df_backtest.loc[date, 'signal'] = current_signal.iloc[exposition_pca_number - 1]

    if df_backtest.shape[0] < min_sample:  # Only compute signal until reach minimun
        df_backtest.loc[date, 'position'] = 0  # Neutral to the PC
        continue

    # Rolling percentile of the signal
    df_backtest.loc[date, 'signal percentile'] = percentileofscore(a=df_backtest['signal'],  # TODO Could add a window here
                                                                   score=df_backtest.loc[date, 'signal'],
                                                                   kind='mean')

    # Build Portfolios
    if (df_backtest.loc[datem1, 'position'] == 1) or (df_backtest.loc[datem1, 'position'] == -1):
        # If yesterday I had a position, check if the signal percentile is back at 50.
        signm1 = np.sign(df_backtest.loc[datem1, 'signal percentile'] - 50)
        sign = np.sign(df_backtest.loc[date, 'signal percentile'] - 50)
        if sign == signm1:
            # Still on the same side of 50, hold the previous position
            df_backtest.loc[date, 'position'] = df_backtest.loc[datem1, 'position']
        else:
            # Signal changed side from 50. Exit the position
            df_backtest.loc[date, 'position'] = 0

    else:
        # If yesterday I did not have a position, check If I should build one
        if df_backtest.loc[date, 'signal percentile'] >= (100 - entry_bound):
            # Build a short position, invert the quantities
            df_backtest.loc[date, 'position'] = -1
        elif df_backtest.loc[date, 'signal percentile'] <= entry_bound:
            # Build a long position
            df_backtest.loc[date, 'position'] = 1
        else:
            # stay out
            df_backtest.loc[date, 'position'] = 0


    # Today's PnL
    # filter_contracts = df_raw['contract'].isin(current_portfolios['contracts'].values)
    # filter_date = df_raw['reference_date'] == date
    # pnl_contracts = df_raw[filter_date & filter_contracts].sort_values('du')['pnl'].values
    # pnl_today = current_portfolios['quantities'].values.dot(pnl_contracts)
    # df_backtest.loc[date, 'notional'] = df_backtest.loc[datem1, 'notional'] + pnl_today
    #
    # # Today's Signal
    # next_roll_date = pd.to_datetime(dates2loop[0]) + pd.offsets.Day(holding_period)
    # # Roll
    # if pd.to_datetime(date) >= next_roll_date:
    #     current_portfolios, current_signal = get_portfolio(datem1, exposition_pcadv01, window_size)
    #     df_backtest.loc[date, 'signal'] = current_signal.iloc[exposition_pca_number - 1]
    #     df_backtest.loc[date, ['du 1', 'du 2', 'du 3']] = current_portfolios.index.sort_values()
    #     next_roll_date = pd.to_datetime(date) + pd.offsets.Day(holding_period)

# df_backtest['notional'] = 100 * df_backtest['notional'] / df_backtest['notional'].iloc[0]

# ax1 = df_backtest['signal'].plot()
# ax2 = ax1.twinx()
# ax2.spines['right'].set_position(('axes', 1.0))
# df_backtest['signal percentile'].plot(ax=ax2, color='orange')
# plt.show()

df_backtest.to_clipboard()

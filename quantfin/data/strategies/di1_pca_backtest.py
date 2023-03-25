import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
from quantfin.data import DROPBOX
from sklearn.decomposition import PCA
from scipy.stats import percentileofscore

pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 250)
tic = time()

# User defined parameters
min_sample = 21
start_date = '2007-01-01'
# TODO add full PCA override
exposition_pca_number = 2
exposition_pcadv01 = [0, 100, 0]
entry_bound = 5  # Must be below 50 # TODO Hyper
writer = pd.ExcelWriter(DROPBOX.joinpath('DI1 PCA Backtest.xlsx'))

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
df_dv01 = - df_dv01

# ===========================
# ===== Full sample PCA =====
# ===========================
pca_full = PCA(n_components=3)
pca_full.fit(df_curve.values)

df_var_full = pd.DataFrame(data=pca_full.explained_variance_ratio_)
df_loadings_full = pd.DataFrame(data=pca_full.components_.T,
                                columns=['PC 1', 'PC 2', 'PC 3'],
                                index=df_curve.columns)
df_mean_full = pd.DataFrame(data=pca_full.mean_, index=df_curve.columns,
                            columns=['Médias'])
df_pca_full = pd.DataFrame(data=pca_full.transform(df_curve.values),
                           columns=['PC 1', 'PC 2', 'PC 3'],
                           index=df_curve.index)

signal_full = np.sign(df_loadings_full.iloc[0])
df_loadings_full = df_loadings_full * signal_full
df_pca_full = df_pca_full * signal_full

df_pca_full.to_excel(writer, 'PCA Full')
df_loadings_full.to_excel(writer, 'Loadings Full')
df_var_full.to_excel(writer, 'Variance Full')


# ====================
# ===== Backtest =====
# ====================
# TODO montar o backtest dos 3 PCAs baseado no df_pca_roll
# Function that selects the portfolio for a given date
def get_portfolio(current_date, pcadv01):
    current_dt = pd.to_datetime(current_date)

    pca = PCA(n_components=3)
    data = df_curve.loc[:current_dt].values
    pca.fit(data)

    current_loadings = pd.DataFrame(data=pca.components_,
                                    index=range(1, 4),
                                    columns=df_curve.columns).T
    current_pca = pd.DataFrame(data=pca.transform(df_curve.values),
                               columns=range(1, 4),
                               index=df_curve.index)

    signal = np.sign(current_loadings.iloc[0])
    current_loadings = current_loadings * signal
    current_pca = current_pca * signal

    available_maturities = df_dv01.loc[current_date].dropna().index
    available_maturities = available_maturities[df_curve.columns.min() <= available_maturities]
    available_maturities = available_maturities[df_curve.columns.max() >= available_maturities]

    aux_pcadv01 = current_loadings.loc[available_maturities]
    aux_pcadv01 = aux_pcadv01.multiply(df_dv01.loc[current_date, available_maturities], axis=0)
    vertices_du = [aux_pcadv01.index[5], aux_pcadv01.idxmin()[3], aux_pcadv01.idxmax()[3]]

    if len(set(vertices_du)) != 3:
        vertices_du = [aux_pcadv01.index[3], aux_pcadv01.idxmin()[3], aux_pcadv01.index[-1]]

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


# Begin Backtest
df_backtest = pd.DataFrame()
dates2loop = df_curve.index[df_curve.index >= start_date]
df_backtest.loc[dates2loop[0], 'position'] = 0

current_portfolios, current_signal = get_portfolio(current_date=dates2loop[0],
                                                   pcadv01=exposition_pcadv01)
df_backtest.loc[dates2loop[0], 'notional'] = (current_portfolios['quantities'].abs() * current_portfolios['pu']).sum()

dates2loop = list(zip(dates2loop[1:], dates2loop[:-1]))
for date, datem1 in tqdm(dates2loop, 'Backtesting'):

    # Signal and portfolio will be computed everyday
    current_portfolios, current_signal = get_portfolio(datem1, exposition_pcadv01)
    df_backtest.loc[date, 'signal'] = current_signal[f'PCA {exposition_pca_number}']

    if df_backtest.shape[0] < min_sample:  # assures min_sample observations before rebalances
        df_backtest.loc[date, 'position'] = 0  # Neutral to the PC
        df_backtest.loc[date, 'notional'] = df_backtest.loc[datem1, 'notional']
        continue
#
    # Rolling percentile of the signal
    signal_percentile = percentileofscore(a=df_backtest['signal'].dropna(),
                                          score=df_backtest.loc[date, 'signal'],
                                          kind='weak')
    df_backtest.loc[date, 'signal percentile'] = signal_percentile

    # Build Position
    if (df_backtest.loc[datem1, 'position'] == 1) or (df_backtest.loc[datem1, 'position'] == -1):
        # If yesterday I had a position, check if the signal percentile is back at 50
        signm1 = np.sign(df_backtest.loc[datem1, 'signal percentile'] - 50)
        sign = np.sign(df_backtest.loc[date, 'signal percentile'] - 50)
        if sign == signm1:
            # Still on the same side of 50, hold the previous position
            df_backtest.loc[date, 'position'] = df_backtest.loc[datem1, 'position']
        else:
            # Signal changed side from 50. Exit the position
            df_backtest.loc[date, 'position'] = 0

    else:  # TODO add exit bound
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
    if np.abs(df_backtest.loc[date, 'position']) == 1:  # If I am positioned
        filter_contracts = df_raw['contract'].isin(current_portfolios['contracts'].values)
        filter_date = df_raw['reference_date'] == date
        pnl_contracts = df_raw[filter_date & filter_contracts].sort_values('du')['pnl'].values
        pnl_today = (df_backtest.loc[date, 'position'] * current_portfolios['quantities']).values.dot(pnl_contracts)
        df_backtest.loc[date, ['du 1', 'du 2', 'du 3']] = current_portfolios.index.sort_values()
        df_backtest.loc[date, ['q 1', 'q 2', 'q 3']] = current_portfolios['quantities'].values
    else:  # If I am not positioned
        pnl_today = 0
        df_backtest.loc[date, ['du 1', 'du 2', 'du 3']] = [0, 0, 0]
        df_backtest.loc[date, ['q 1', 'q 2', 'q 3']] = [0, 0, 0]

    df_backtest.loc[date, 'pnl'] = pnl_today
    df_backtest.loc[date, 'notional'] = df_backtest.loc[datem1, 'notional'] + pnl_today
    # TODO add cost

df_backtest.to_excel(writer, 'Backtest')

writer.save()
toc = time()
print(round((toc - tic) / 60, 1), 'minutes')

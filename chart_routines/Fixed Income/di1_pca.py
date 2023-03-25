import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.decomposition import PCA
from quantfin.data import DROPBOX
from pathlib import Path
import getpass

# User defined parameters
start_date = '2007-01-01'
exposition_pcadv01 = [100, 0, 0]
exposition_pca_number = 1
username = getpass.getuser()
save_path = Path(f'/Users/{username}/Dropbox/Aulas/Insper - Renda Fixa/2023/figures')

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
df_curve = df_raw.pivot(index='reference_date', columns='du', values='rate')
df_curve = df_curve.interpolate(axis=1, method='cubic', limit_area='inside')
df_curve = df_curve.dropna(how='any', axis=1)
df_curve.index = pd.to_datetime(df_curve.index)

# Organize DV01
df_dv01 = df_raw.pivot(index='reference_date', columns='du', values='dv01')
df_dv01.index = pd.to_datetime(df_dv01.index)
# df_dv01 = - df_dv01


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

signal = np.sign(df_loadings_full.iloc[-1])
df_loadings_full = df_loadings_full * signal
df_pca_full = df_pca_full * signal


# ==================
# ===== Charts =====
# ==================
# Curve dynamics
df_plot = df_curve[[252*a for a in range(1, 10)]] * 100

plt.figure(figsize=(7, 7 * (9 / 16)))
plt.plot(df_plot)

last_date = df_curve.index[-1]
plt.title(f"Yields of the DI Curve as of {last_date.strftime('%d/%b/%y')}")


ax = plt.gca()
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.tick_params(rotation=90, axis='x')
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.legend(df_plot.columns, frameon=True, loc='best', ncol=2)

plt.tight_layout()
plt.savefig(save_path.joinpath('DI Yields.pdf'))
plt.show()
plt.close()


# Time Series of the PCs
df_plot = df_pca_full

plt.figure(figsize=(7, 7 * (9 / 16)))
plt.plot(df_plot)


last_date = df_curve.index[-1]
plt.title(f"Principal Components of the DI Curve as of {last_date.strftime('%d/%b/%y')}")


ax = plt.gca()
ax.axhline(0, color='black', linewidth=0.5)
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.tick_params(rotation=90, axis='x')
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.legend(df_plot.columns, frameon=True, loc='best')

plt.tight_layout()
plt.savefig(save_path.joinpath('DI Principal Components.pdf'))
plt.show()
plt.close()

# Factor Loadings
df_plot = df_loadings_full * 100

plt.figure(figsize=(7, 7 * (9 / 16)))
plt.plot(df_plot)
plt.xlabel('Maturity (DU)')

last_date = df_curve.index[-1]
plt.title(f"Factor Loadings of the DI Curve as of {last_date.strftime('%d/%b/%y')}")


ax = plt.gca()
ax.axhline(0, color='black', linewidth=0.5)
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.legend(df_plot.columns, frameon=True, loc='best')

plt.tight_layout()
plt.savefig(save_path.joinpath('DI Factor Loadings.pdf'))
plt.show()
plt.close()

# Explained Variance
df_plot = df_var_full

plt.figure(figsize=(7, 7 * (9 / 16)))
plt.bar([f'PC{a}' for a in range(1, 4)], df_plot.iloc[:, 0].values)
plt.plot(df_plot.iloc[:, 0].cumsum(), color='orange')

last_date = df_curve.index[-1]
plt.title(f"Explained Variance of the DI Curve as of {last_date.strftime('%d/%b/%y')}")


ax = plt.gca()
ax.axhline(0, color='black', linewidth=0.5)
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig(save_path.joinpath('DI Explained Variance.pdf'))
plt.show()
plt.close()


# ====================
# ===== Backtest =====
# ====================
def get_portfolio(current_date, pcadv01):
    """
    Function that Selects the portfolio for a given set of signals
    """
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

    signal = np.sign(current_loadings.iloc[-1])
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


# Start
df_backtest = pd.DataFrame()
dates2loop = df_curve.index[df_curve.index >= start_date]

current_portfolios, current_signal = get_portfolio(current_date=dates2loop[0],
                                                   pcadv01=exposition_pcadv01)

df_backtest.loc[dates2loop[0], 'notional'] = (current_portfolios['quantities'].abs() * current_portfolios['pu']).sum()

dates2loop = list(zip(dates2loop[1:], dates2loop[:-1]))
for date, datem1 in tqdm(dates2loop, 'Backtesting'):

    # Signal and portfolio will be computed everyday
    current_portfolios, current_signal = get_portfolio(datem1, exposition_pcadv01)
    df_backtest.loc[date, 'signal'] = current_signal[f'PCA {exposition_pca_number}']

    # Build Position
    filter_contracts = df_raw['contract'].isin(current_portfolios['contracts'].values)
    filter_date = df_raw['reference_date'] == date
    pnl_contracts = df_raw[filter_date & filter_contracts].sort_values('du')['pnl'].values
    pnl_today = current_portfolios['quantities'].values.dot(pnl_contracts)

    df_backtest.loc[date, ['du 1', 'du 2', 'du 3']] = current_portfolios.index.sort_values()
    df_backtest.loc[date, ['q 1', 'q 2', 'q 3']] = current_portfolios['quantities'].values
    df_backtest.loc[date, 'pnl'] = pnl_today
    df_backtest.loc[date, 'notional'] = df_backtest.loc[datem1, 'notional'] + pnl_today

df_backtest.to_clipboard()

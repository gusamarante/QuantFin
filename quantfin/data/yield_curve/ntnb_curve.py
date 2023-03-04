from scipy.optimize import minimize, Bounds
from quantfin.calendars import DayCounts
from scipy.interpolate import interp1d
from quantfin.data import DROPBOX
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)
dc = DayCounts(dc='bus/252', calendar='anbima')


# Function to generate ntnb cashflows
def ntnb_cashflows(reference_date, maturity_date, vna):
    mat_year = maturity_date.year

    if mat_year % 2 == 0:  # Se o ano é par, cupon em fev e ago, vence em ago
        dates = pd.date_range(start='1980-02-15', end=maturity_date, freq='12SM')

    else:  # se o ano é ímpar, cupom em mai e nov, vence em mai
        dates = pd.date_range(start='1980-05-15', end=maturity_date, freq='12SM')

    dates = dates[dates >= reference_date]
    dates = dc.following(dates)
    n_cf = len(dates)
    dus = dc.days(reference_date, dates)

    coupons = 1.06 ** 0.5 - 1 * np.ones(n_cf)
    coupons[-1] = coupons[-1] + 1
    coupons = coupons * vna

    cashflows = pd.DataFrame(index=dus, data={'cashflow': coupons})

    return cashflows


# ===== optimization =====
def bootstrapp(cashflows, prices):
    # TODO Documentation

    # Find the DUs that we can change
    du_dof = cashflows.idxmax().values

    def objective_function(disc):
        dus = np.insert(du_dof, 0, 0)  # add the first value, which will be fixed at zero
        disc = np.insert(disc, 0, 1)  # add the first value, which will be fixed at one
        f = interp1d(dus, np.log(disc))  # Interpolation of the log of disccounts
        disc = pd.Series(index=cashflows.index, data=np.exp(f(cashflows.index)))  # Populate the discounts to a series
        sum_dcf = cashflows.multiply(disc, axis=0).sum()  # get the sum of discounted cashflows
        erros = prices.subtract(sum_dcf, axis=0)  # Difference between actual prices and sum of DCF
        erro_total = (erros ** 2).sum()  # Sum of squarred errors

        try:
            erro_total = erro_total.values[0]
        except AttributeError:
            erro_total = erro_total

        return erro_total

    # Run optimization
    # Initial gues for the vector of disccounts
    init_discount = 0.8 * np.ones(len(du_dof))
    res = minimize(fun=objective_function,
                   x0=init_discount,
                   method=None,
                   tol=1e-16,
                   options={'disp': False})

    dus = np.insert(du_dof, 0, 0)  # add the first value, which will be fixed at zero
    discount = np.insert(res.x, 0, 1)  # add the first value, which will be fixed at one
    f = interp1d(dus, np.log(discount))  # Interpolation of the log of disccounts
    discount = pd.Series(index=cashflows.index, data=np.exp(f(cashflows.index)))

    curve = (1 / discount) ** (252 / discount.index) - 1

    return curve


# Read the Data
last_year = 2023
raw_data = pd.DataFrame()

for year in tqdm(range(2003, last_year + 1), 'Reading files'):
    aux = pd.read_csv(DROPBOX.joinpath(f'trackers/dados_ntnb {year}.csv'), sep=';')
    raw_data = pd.concat([raw_data, aux])

raw_data = raw_data.drop(['Unnamed: 0', 'index'], axis=1)
raw_data['reference date'] = pd.to_datetime(raw_data['reference date'])
raw_data['maturity'] = pd.to_datetime(raw_data['maturity'])

# Iterate for every date
dates2loop = raw_data['reference date'].drop_duplicates().sort_values()
ano = 2022
dates2loop = dates2loop[dates2loop >= f'{ano}-01-01']
dates2loop = dates2loop[dates2loop <= f'{ano}-12-31']

df_yield_curve = pd.DataFrame()
for today in tqdm(dates2loop, 'Bootstrapping'):

    current_bonds = raw_data[raw_data['reference date'] == today].sort_values('du')

    df_cashflows = pd.DataFrame()
    for bond in current_bonds['maturity']:
        aux = ntnb_cashflows(reference_date=today,
                             maturity_date=bond,
                             vna=current_bonds['vna'].max())
        aux = aux.rename({'cashflow': bond}, axis=1)
        df_cashflows = pd.concat([df_cashflows, aux], axis=1)

    df_cashflows = df_cashflows.sort_index()
    df_cashflows = df_cashflows.fillna(0)
    prices = current_bonds[['maturity', 'price']].set_index('maturity')

    yield_curve = bootstrapp(df_cashflows, prices)
    yield_curve = yield_curve.to_frame(today)
    yield_curve = yield_curve.melt(ignore_index=False).reset_index()
    yield_curve = yield_curve.rename({'index': 'du',
                                      'variable': 'reference_date',
                                      'value': 'yield'},
                                     axis=1)

    df_yield_curve = pd.concat([df_yield_curve, yield_curve], axis=0)

df_yield_curve = df_yield_curve.dropna()
df_yield_curve.to_csv(f'/Users/gusamarante/Library/CloudStorage/Dropbox/Personal Portfolio/curves/curva_zero_ntnb_{ano}.csv')

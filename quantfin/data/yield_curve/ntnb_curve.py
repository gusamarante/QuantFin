from scipy.optimize import minimize, Bounds
from quantfin.calendars import DayCounts
from scipy.interpolate import interp1d
from quantfin.data import DROPBOX
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)
dc = DayCounts(dc='bus/252', calendar='anbima')

# Read the Data
last_year = 2022
raw_data = pd.DataFrame()

for year in tqdm(range(2003, last_year + 1), 'Reading files'):
    aux = pd.read_csv(DROPBOX.joinpath(f'trackers/dados_ntnb {year}.csv'), sep=';')
    raw_data = pd.concat([raw_data, aux])

raw_data = raw_data.drop(['Unnamed: 0', 'index'], axis=1)
raw_data['reference date'] = pd.to_datetime(raw_data['reference date'])
raw_data['maturity'] = pd.to_datetime(raw_data['maturity'])


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


# EXAMPLE - Single date
today = raw_data['reference date'].max()
current_bonds = raw_data[raw_data['reference date'] == today].sort_values('du')

df = pd.DataFrame()
for bond in current_bonds['maturity']:

    aux = ntnb_cashflows(reference_date=today,
                         maturity_date=bond,
                         vna=current_bonds['vna'].max())
    aux = aux.rename({'cashflow': bond}, axis=1)
    df = pd.concat([df, aux], axis=1)

df = df.sort_index()
df = df.fillna(0)


# ===== optimization =====
def bootstrapp(cashflows, prices):

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
                   options={'ftol': 1e-16,
                            'disp': False})

    dus = np.insert(du_dof, 0, 0)  # add the first value, which will be fixed at zero
    discount = np.insert(res.x, 0, 1)  # add the first value, which will be fixed at one
    f = interp1d(dus, np.log(discount))  # Interpolation of the log of disccounts
    discount = pd.Series(index=cashflows.index, data=np.exp(f(cashflows.index)))

    curve = (1 / discount) ** (252 / discount.index) - 1

    print(res.fun)

    return curve


prices = current_bonds[['maturity', 'price']].set_index('maturity')
# print(bootstrapp(df, prices))
yield_curve = bootstrapp(df, prices)
yield_curve.plot()
plt.show()

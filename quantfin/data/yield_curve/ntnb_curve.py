from quantfin.data import DROPBOX
from quantfin.calendars import DayCounts
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

    df = pd.concat([df, aux], axis=1)

df = df.sort_index()
df = df.fillna(0)

CF = df.values.T
P = current_bonds['price'].values



print(df)

import matplotlib.pyplot as plt
from quantfin.data import DROPBOX
from tqdm import tqdm
from time import time
import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)

tic = time()

# User defined parameters
notional_start = 100
# start_date = '2022-01-01'

# Set up
ntnb = pd.read_csv(DROPBOX.joinpath('trackers/dados_ntnb.csv'), sep=';')
ntnb['reference date'] = pd.to_datetime(ntnb['reference date'])
dates2loop = pd.to_datetime(ntnb['reference date'].unique())
# dates2loop = dates2loop[dates2loop >= start_date]
df_bt = pd.DataFrame()

# First date
aux_data = ntnb[ntnb['reference date'] == dates2loop[0]].set_index('bond code')
aux_data = aux_data.sort_values('du')
filter_du = aux_data['du'] >= 60
aux_data = aux_data[filter_du]

current_bond = aux_data.index[0]
df_bt.loc[dates2loop[0], 'bond'] = current_bond
df_bt.loc[dates2loop[0], 'du'] = aux_data.iloc[0]['du']
df_bt.loc[dates2loop[0], 'quantity'] = notional_start / (aux_data.iloc[0]['price'] + aux_data.iloc[0]['bidask spread'] / 2)
df_bt.loc[dates2loop[0], 'price'] = aux_data.iloc[0]['price']
df_bt.loc[dates2loop[0], 'Notional'] = df_bt.loc[dates2loop[0], 'quantity'] * df_bt.loc[dates2loop[0], 'price']

for date, datem1 in tqdm(zip(dates2loop[1:], dates2loop[:-1]), 'Backtesting NTN-B curta'):

    if date == pd.to_datetime('2019-02-19'):
        a = 1

    # get available bonds today
    aux_data = ntnb[ntnb['reference date'] == date].set_index('bond code')
    aux_data = aux_data.sort_values('du')
    filter_du = aux_data['du'] >= 60
    aux_data_get = aux_data[filter_du]

    # check if the shortest bond changed or not
    new_bond = aux_data_get.index[0]
    if new_bond == current_bond:  # still the same, hold the position, check for coupons
        df_bt.loc[date, 'bond'] = current_bond
        df_bt.loc[date, 'du'] = aux_data.loc[current_bond, 'du']
        df_bt.loc[date, 'quantity'] = df_bt.loc[datem1, 'quantity'] * (1 + aux_data.loc[current_bond, 'coupon'] / (aux_data.loc[current_bond, 'price'] + aux_data.loc[current_bond, 'bidask spread'] / 2))
        df_bt.loc[date, 'price'] = aux_data.loc[current_bond, 'price']
        df_bt.loc[date, 'Notional'] = df_bt.loc[date, 'quantity'] * df_bt.loc[date, 'price']

    else:  # new longer bond, roll the position
        df_bt.loc[date, 'bond'] = new_bond
        df_bt.loc[date, 'du'] = aux_data.loc[new_bond, 'du']
        sellvalue = df_bt.loc[datem1, 'quantity'] * (aux_data.loc[current_bond, 'price'] - aux_data.loc[current_bond, 'bidask spread'] / 2)
        df_bt.loc[date, 'quantity'] = (sellvalue + df_bt.loc[datem1, 'quantity'] * aux_data.loc[current_bond, 'coupon']) / (aux_data.loc[new_bond, 'price'] + aux_data.loc[new_bond, 'bidask spread'] / 2)
        df_bt.loc[date, 'price'] = aux_data.loc[new_bond, 'price']
        df_bt.loc[date, 'Notional'] = df_bt.loc[date, 'quantity'] * df_bt.loc[date, 'price']
        current_bond = new_bond

df_bt.to_csv(DROPBOX.joinpath('trackers/ntnb_curta.csv'), sep=';')
minutes = round((time() - tic) / 60, 2)
print('NTNB Curta took', minutes, 'minutes')

from quantfin.data import DROPBOX
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)

last_year = 2023
tic = time()

# User defined parameters
notional_start = 100
start_date = '2006-01-01'

# Set up
lft = pd.DataFrame()

for year in tqdm(range(2003, last_year + 1), 'Reading files'):
    aux = pd.read_csv(DROPBOX.joinpath(f'trackers/dados_lft {year}.csv'), sep=';')
    lft = pd.concat([lft, aux])

lft['reference date'] = pd.to_datetime(lft['reference date'])
dates2loop = pd.to_datetime(lft['reference date'].unique())
dates2loop = dates2loop[dates2loop >= start_date]
df_bt = pd.DataFrame()

# First date
aux_data = lft[lft['reference date'] == dates2loop[0]].set_index('bond code')
aux_data = aux_data.sort_values('du')
filter_du = aux_data['du'] >= 60
aux_data = aux_data[filter_du]

current_bond = aux_data.index[0]
df_bt.loc[dates2loop[0], 'bond'] = current_bond
df_bt.loc[dates2loop[0], 'du'] = aux_data.loc[current_bond, 'du']
df_bt.loc[dates2loop[0], 'quantity'] = notional_start / (aux_data.loc[current_bond, 'price'] + aux_data.loc[current_bond, 'bidask spread'] / 2)
df_bt.loc[dates2loop[0], 'price'] = aux_data.loc[current_bond, 'price']
df_bt.loc[dates2loop[0], 'Notional'] = df_bt.loc[dates2loop[0], 'quantity'] * df_bt.loc[dates2loop[0], 'price']

for date, datem1 in tqdm(zip(dates2loop[1:], dates2loop[:-1]), 'Backtesting LFT curta'):

    # get available bonds today
    aux_data = lft[lft['reference date'] == date].set_index('bond code')
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

    else:  # new shorter bond, roll the position
        df_bt.loc[date, 'bond'] = new_bond
        df_bt.loc[date, 'du'] = aux_data.loc[new_bond, 'du']
        sellvalue = df_bt.loc[datem1, 'quantity'] * (aux_data.loc[current_bond, 'price'] - aux_data.loc[current_bond, 'bidask spread'] / 2)
        df_bt.loc[date, 'quantity'] = (sellvalue + df_bt.loc[datem1, 'quantity'] * aux_data.loc[current_bond, 'coupon']) / (aux_data.loc[new_bond, 'price'] + aux_data.loc[new_bond, 'bidask spread'] / 2)
        df_bt.loc[date, 'price'] = aux_data.loc[new_bond, 'price']
        df_bt.loc[date, 'Notional'] = df_bt.loc[date, 'quantity'] * df_bt.loc[date, 'price']
        current_bond = new_bond

df_bt.to_csv(DROPBOX.joinpath('trackers/lft_curta.csv'), sep=';')
minutes = round((time() - tic) / 60, 2)
print('LFT Curta took', minutes, 'minutes')

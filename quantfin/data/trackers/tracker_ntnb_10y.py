from quantfin.data import DROPBOX
from tqdm import tqdm
from time import time
import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)

desired_duration = 10  # in years (its going to be a little more)
rebalance_cost = 0.0015
rebalance_window = 3  # in months
last_year = 2022
tic = time()

# User defined parameters
notional_start = 100
start_date = '2006-01-01'

# Set up
ntnb = pd.DataFrame()

for year in tqdm(range(2003, last_year + 1), 'Reading files'):
    aux = pd.read_csv(DROPBOX.joinpath(f'trackers/dados_ntnb {year}.csv'), sep=';')
    ntnb = pd.concat([ntnb, aux])

ntnb['reference date'] = pd.to_datetime(ntnb['reference date'])
dates2loop = pd.to_datetime(ntnb['reference date'].unique())
dates2loop = dates2loop[dates2loop >= start_date]
df_bt = pd.DataFrame()

# First date
aux_data = ntnb[ntnb['reference date'] == dates2loop[0]].set_index('bond code')
aux_data = aux_data.sort_values('du')

dur_idx = aux_data['duration'].searchsorted(desired_duration)
a = aux_data['duration'].iloc[[dur_idx - 1, dur_idx]].values
x = (desired_duration - a[1]) / (a[0] - a[1])  # Ammount of bond 1
current_bond1, current_bond2 = aux_data['duration'].iloc[[dur_idx - 1, dur_idx]].index
df_bt.loc[dates2loop[0], 'bond 1'] = current_bond1
df_bt.loc[dates2loop[0], 'bond 2'] = current_bond2
df_bt.loc[dates2loop[0], 'du 1'] = aux_data.loc[current_bond1, 'du']
df_bt.loc[dates2loop[0], 'du 2'] = aux_data.loc[current_bond2, 'du']
df_bt.loc[dates2loop[0], 'quantity 1'] = x * notional_start / (aux_data.loc[current_bond1, 'price'] + aux_data.loc[current_bond1, 'bidask spread'] / 2)
df_bt.loc[dates2loop[0], 'quantity 2'] = (1 - x) * notional_start / (aux_data.loc[current_bond2, 'price'] + aux_data.loc[current_bond2, 'bidask spread'] / 2)
df_bt.loc[dates2loop[0], 'price 1'] = aux_data.loc[current_bond1, 'price']
df_bt.loc[dates2loop[0], 'price 2'] = aux_data.loc[current_bond2, 'price']
df_bt.loc[dates2loop[0], 'Notional'] = df_bt.loc[dates2loop[0], 'quantity 1'] * df_bt.loc[dates2loop[0], 'price 1'] \
                                     + df_bt.loc[dates2loop[0], 'quantity 2'] * df_bt.loc[dates2loop[0], 'price 2']

next_rebalance_date = dates2loop[0] + pd.DateOffset(months=rebalance_window)

for date, datem1 in tqdm(zip(dates2loop[1:], dates2loop[:-1]), 'Backtesting NTN-B 10y'):

    # get available bonds today
    aux_data = ntnb[ntnb['reference date'] == date].set_index('bond code')
    aux_data = aux_data.sort_values('du')

    if date < next_rebalance_date:  # still behind the rebalance, MtM
        df_bt.loc[date, 'bond 1'] = current_bond1
        df_bt.loc[date, 'bond 2'] = current_bond2
        df_bt.loc[date, 'du 1'] = aux_data.loc[current_bond1, 'du']
        df_bt.loc[date, 'du 2'] = aux_data.loc[current_bond2, 'du']
        df_bt.loc[date, 'quantity 1'] = df_bt.loc[datem1, 'quantity 1'] * (1 + aux_data.loc[current_bond1, 'coupon'] / (aux_data.loc[current_bond1, 'price'] + aux_data.loc[current_bond1, 'bidask spread'] / 2))
        df_bt.loc[date, 'quantity 2'] = df_bt.loc[datem1, 'quantity 2'] * (1 + aux_data.loc[current_bond2, 'coupon'] / (aux_data.loc[current_bond2, 'price'] + aux_data.loc[current_bond2, 'bidask spread'] / 2))
        df_bt.loc[date, 'price 1'] = aux_data.loc[current_bond1, 'price']
        df_bt.loc[date, 'price 2'] = aux_data.loc[current_bond2, 'price']
        df_bt.loc[date, 'Notional'] = df_bt.loc[date, 'quantity 1'] * df_bt.loc[date, 'price 1'] + df_bt.loc[date, 'quantity 2'] * df_bt.loc[date, 'price 2']

    else:  # past rebalance, recompute the weights
        # TODO add rebalance cost
        dur_idx = aux_data['duration'].searchsorted(desired_duration)
        a = aux_data['duration'].iloc[[dur_idx - 1, dur_idx]].values
        x = (desired_duration - a[1]) / (a[0] - a[1])  # Ammount of bond 1
        new_bond1, new_bond2 = aux_data['duration'].iloc[[dur_idx - 1, dur_idx]].index
        df_bt.loc[date, 'bond 1'] = new_bond1
        df_bt.loc[date, 'bond 2'] = new_bond2
        df_bt.loc[date, 'du 1'] = aux_data.loc[new_bond1, 'du']
        df_bt.loc[date, 'du 2'] = aux_data.loc[new_bond2, 'du']

        # TODO add rebalance cost of selling
        sellvalue = df_bt.loc[datem1, 'quantity 1'] * (aux_data.loc[current_bond1, 'price'])
        sellvalue = sellvalue + df_bt.loc[datem1, 'quantity 2'] * (aux_data.loc[current_bond2, 'price'])
        sellvalue = sellvalue + df_bt.loc[datem1, 'quantity 1'] * aux_data.loc[current_bond1, 'coupon'] + df_bt.loc[datem1, 'quantity 2'] * aux_data.loc[current_bond2, 'coupon']

        # TODO add rebalance cost of buying
        df_bt.loc[date, 'quantity 1'] = x * sellvalue / (aux_data.loc[new_bond1, 'price'])
        df_bt.loc[date, 'quantity 2'] = (1 - x) * sellvalue / (aux_data.loc[new_bond2, 'price'] )

        df_bt.loc[date, 'price 1'] = aux_data.loc[new_bond1, 'price']
        df_bt.loc[date, 'price 2'] = aux_data.loc[new_bond2, 'price']

        df_bt.loc[date, 'Notional'] = df_bt.loc[date, 'quantity 1'] * df_bt.loc[date, 'price 1'] \
                                    + df_bt.loc[date, 'quantity 2'] * df_bt.loc[date, 'price 2']

        current_bond1, current_bond2 = new_bond1, new_bond2

df_bt.to_csv(DROPBOX.joinpath('trackers/ntnb_10y.csv'), sep=';')
minutes = round(time() - tic, 2)
print('NTNB 10y took', minutes, 'seconds')


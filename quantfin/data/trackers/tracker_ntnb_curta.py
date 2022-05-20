import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from titulospublicos import NTNB
from time import time

tic = time()

# User defined parameters
notional_start = 100
start_date = '2022-01-01'

# Set up
ntnb = NTNB()
dates2loop = pd.to_datetime(ntnb.time_menu())
# dates2loop = dates2loop[dates2loop >= start_date]
df_bt = pd.DataFrame()

# First date
bond_codes = ntnb.market_menu(dates2loop[0])
aux_data = pd.DataFrame(data={'yield': [ntnb.implied_yield(dates2loop[0], bond) for bond in bond_codes],
                              'price': [ntnb.theor_price(dates2loop[0], bond, ntnb.implied_yield(dates2loop[0], bond)) for bond in bond_codes],
                              'bidask spread': [ntnb.bidofferspread(dates2loop[0], bond) for bond in bond_codes],
                              'du': [ntnb.du2maturity(dates2loop[0], bond) for bond in bond_codes],
                              'coupon': [ntnb.cash_payment(dates2loop[0], bond) for bond in bond_codes]},
                        index=bond_codes)
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

    # get available bonds today
    bond_codes = ntnb.market_menu(date)
    aux_data = pd.DataFrame(data={'yield': [ntnb.implied_yield(date, bond) for bond in bond_codes],
                                  'price': [ntnb.theor_price(date, bond, ntnb.implied_yield(date, bond)) for bond in bond_codes],
                                  'bidask spread': [ntnb.bidofferspread(date, bond) for bond in bond_codes],
                                  'du': [ntnb.du2maturity(date, bond) for bond in bond_codes],
                                  'coupon': [ntnb.cash_payment(date, bond) for bond in bond_codes]},
                            index=bond_codes)
    aux_data = aux_data.sort_values('du')
    filter_du = aux_data['du'] >= 60
    aux_data = aux_data[filter_du]

    # check if the longest bond changed or not
    new_bond = aux_data.index[0]
    if new_bond == current_bond:  # still the same, hold the position, check for coupons
        df_bt.loc[date, 'bond'] = current_bond
        df_bt.loc[date, 'du'] = aux_data.iloc[0]['du']
        df_bt.loc[date, 'quantity'] = df_bt.loc[datem1, 'quantity'] * (1 + aux_data.iloc[0]['coupon'] / (aux_data.iloc[0]['price'] + aux_data.iloc[0]['bidask spread'] / 2))
        df_bt.loc[date, 'price'] = aux_data.iloc[0]['price']
        df_bt.loc[date, 'Notional'] = df_bt.loc[date, 'quantity'] * df_bt.loc[date, 'price']

    else:  # new longer bond, roll the position
        df_bt.loc[date, 'bond'] = new_bond
        df_bt.loc[date, 'du'] = aux_data.iloc[0]['du']
        sellvalue = df_bt.loc[datem1, 'quantity'] * (ntnb.theor_price(date, current_bond, ntnb.implied_yield(date, current_bond)) - ntnb.bidofferspread(date, current_bond) / 2)
        df_bt.loc[date, 'quantity'] = (sellvalue + df_bt.loc[datem1, 'quantity'] * ntnb.cash_payment(date, current_bond)) / (aux_data.iloc[0]['price'] + aux_data.iloc[0]['bidask spread'] / 2)
        df_bt.loc[date, 'price'] = aux_data.iloc[0]['price']
        df_bt.loc[date, 'Notional'] = df_bt.loc[date, 'quantity'] * df_bt.loc[date, 'price']
        current_bond = new_bond

df_bt.to_csv(r'C:\Users\gamarante\Dropbox\Personal Portfolio\trackers\ntnb_curta.csv', sep=';')
minutes = round((time() - tic) / 60, 2)
print('NTNB Curta took', minutes, 'minutes')
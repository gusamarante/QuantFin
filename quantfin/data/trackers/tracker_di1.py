"""
Builds the total return indexes for the DI1
"""
from quantfin.data import DROPBOX
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import pandas as pd

tic = time()  # Time the run

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)

# User defined parameters
desired_duration = [63, 126] + [252*x for x in range(1, 10)]  # in DU
rebalance_window = 2  # in months
last_year = 2023
notional_start = 100
start_date = '2007-01-01'

# Read the DIs
di1 = pd.DataFrame()
for year in tqdm(range(2006, 2023 + 1), 'Reading Files'):
    aux = pd.read_csv(DROPBOX.joinpath(f'trackers/dados_di1 {year}.csv'),
                      sep=';')
    di1 = pd.concat([di1, aux], axis=0)
di1 = di1.drop(['Unnamed: 0'], axis=1)
di1['reference_date'] = pd.to_datetime(di1['reference_date'])
di1['maturity_date'] = pd.to_datetime(di1['maturity_date'])

# Set up
dates2loop = pd.to_datetime(di1['reference_date'].unique())
dates2loop = dates2loop[dates2loop >= start_date]
df_tracker = pd.DataFrame()  # To save all of the final trackers

# ===== Fixed Duration =====
for dd in desired_duration:
    df_bt = pd.DataFrame()

    # First date
    aux_data = di1[di1['reference_date'] == dates2loop[0]].set_index('contract')
    aux_data = aux_data.sort_values('du')
    # aux_data = aux_data[aux_data['volume'] > 100]  # TODO Change to Open interest
    dur_idx = aux_data['du'].searchsorted(dd)
    a = aux_data['du'].iloc[[dur_idx - 1, dur_idx]].values
    x = (dd - a[1]) / (a[0] - a[1])  # Ammount of contract 1
    current_bond1, current_bond2 = aux_data['du'].iloc[[dur_idx - 1, dur_idx]].index
    df_bt.loc[dates2loop[0], 'bond 1'] = current_bond1
    df_bt.loc[dates2loop[0], 'bond 2'] = current_bond2
    df_bt.loc[dates2loop[0], 'du 1'] = aux_data.loc[current_bond1, 'du']
    df_bt.loc[dates2loop[0], 'du 2'] = aux_data.loc[current_bond2, 'du']
    df_bt.loc[dates2loop[0], 'quantity 1'] = x * notional_start / aux_data.loc[current_bond1, 'theoretical_price']
    df_bt.loc[dates2loop[0], 'quantity 2'] = (1 - x) * notional_start / aux_data.loc[current_bond2, 'theoretical_price']

    notional = df_bt.loc[dates2loop[0], 'quantity 1'] * aux_data.loc[current_bond1, 'theoretical_price']
    notional = notional + df_bt.loc[dates2loop[0], 'quantity 2'] * aux_data.loc[current_bond2, 'theoretical_price']
    df_bt.loc[dates2loop[0], 'Notional'] = notional

    next_rebalance_date = dates2loop[0] + pd.DateOffset(months=rebalance_window)

    # Loop for other dates
    paired_dates = zip(dates2loop[1:], dates2loop[:-1])
    for date, datem1 in tqdm(paired_dates, f'Building DI1 Trackers {dd}du'):

        # get available bonds today
        aux_data = di1[di1['reference_date'] == date].set_index('contract')
        aux_data = aux_data.sort_values('du')
        # aux_data = aux_data[aux_data['volume'] > 100]  # TODO Change to Open interest

        if date < next_rebalance_date:  # still behind the rebalance, MtM
            df_bt.loc[date, 'bond 1'] = current_bond1
            df_bt.loc[date, 'bond 2'] = current_bond2
            df_bt.loc[date, 'du 1'] = aux_data.loc[current_bond1, 'du']
            df_bt.loc[date, 'du 2'] = aux_data.loc[current_bond2, 'du']
            df_bt.loc[date, 'quantity 1'] = df_bt.loc[datem1, 'quantity 1']
            df_bt.loc[date, 'quantity 2'] = df_bt.loc[datem1, 'quantity 2']
            pnl = aux_data.loc[current_bond1, 'pnl'] * df_bt.loc[date, 'quantity 1']
            pnl = pnl + aux_data.loc[current_bond2, 'pnl'] * df_bt.loc[date, 'quantity 2']
            df_bt.loc[date, 'Notional'] = df_bt.loc[datem1, 'Notional'] + pnl

        else:  # past rebalance, recompute the weights
            aux_data_select = aux_data[aux_data['du'] > 21 * rebalance_window]
            dur_idx = aux_data_select['du'].searchsorted(dd)

            if dur_idx == 0:
                x = 1
                new_bond1, new_bond2 = aux_data_select['du'].iloc[[0, 1]].index

            elif dur_idx == len(aux_data_select['du']):
                x = 0
                new_bond1, new_bond2 = aux_data_select['du'].iloc[[-2, -1]].index

            else:
                a = aux_data_select['du'].iloc[[dur_idx - 1, dur_idx]].values
                x = (dd - a[1]) / (a[0] - a[1])  # Ammount of bond 1
                new_bond1, new_bond2 = aux_data_select['du'].iloc[[dur_idx - 1, dur_idx]].index

            df_bt.loc[date, 'bond 1'] = new_bond1
            df_bt.loc[date, 'bond 2'] = new_bond2
            df_bt.loc[date, 'du 1'] = aux_data.loc[new_bond1, 'du']
            df_bt.loc[date, 'du 2'] = aux_data.loc[new_bond2, 'du']

            try:
                # Add half of today's pnl to the notional
                pnl = aux_data.loc[current_bond1, 'pnl'] * df_bt.loc[datem1, 'quantity 1'] * 0.5
                pnl = pnl + aux_data.loc[current_bond2, 'pnl'] * df_bt.loc[datem1, 'quantity 2'] * 0.5
            except KeyError:
                # If contract is too short and not available, assume zero
                pnl = 0

            sell_value = df_bt.loc[datem1, 'Notional'] + pnl  # This is only used to adjust the total size.
            df_bt.loc[date, 'quantity 1'] = x * sell_value / aux_data.loc[new_bond1, 'theoretical_price']
            df_bt.loc[date, 'quantity 2'] = (1 - x) * sell_value / aux_data.loc[new_bond2, 'theoretical_price']

            df_bt.loc[date, 'Notional'] = df_bt.loc[datem1, 'Notional'] + pnl

            current_bond1, current_bond2 = new_bond1, new_bond2
            next_rebalance_date = date + pd.DateOffset(months=rebalance_window)

    # Add tracker to df of trackers
    df_tracker = pd.concat([df_tracker, df_bt['Notional'].rename(dd)], axis=1)


df_tracker.to_csv(DROPBOX.joinpath(f'trackers/trackers di1.csv'))
print(round((time() - tic)/60, 1), 'minutes for DI1 trackers')

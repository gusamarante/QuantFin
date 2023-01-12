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
desired_duration = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8]  # in years
rebalance_window = 3  # in months
last_year = 2023
notional_start = 100
start_date = '2006-01-01'

# Read the Data
ntnf = pd.DataFrame()

for year in tqdm(range(2003, last_year + 1), 'Reading files'):
    aux = pd.read_csv(DROPBOX.joinpath(f'trackers/dados_ntnf {year}.csv'), sep=';')
    ntnf = pd.concat([ntnf, aux])

ntnf['reference date'] = pd.to_datetime(ntnf['reference date'])
ntnf['maturity'] = pd.to_datetime(ntnf['maturity'])
ntnf = ntnf.drop(['Unnamed: 0', 'index'], axis=1)

# Set up
dates2loop = pd.to_datetime(ntnf['reference date'].unique())
dates2loop = dates2loop[dates2loop >= start_date]
df_tracker = pd.DataFrame()  # To save all of the final trackers

# ===== Fixed Duration =====
for dd in desired_duration:
    df_bt = pd.DataFrame()

    # First date
    aux_data = ntnf[ntnf['reference date'] == dates2loop[0]].set_index('bond code')
    aux_data = aux_data.sort_values('du')

    dur_idx = aux_data['duration'].searchsorted(dd)

    if dur_idx == 0:  # The shortest maturity already has a higher duration
        x = 1
        current_bond1, current_bond2 = aux_data['duration'].iloc[[0, 1]].index
    elif dur_idx == aux_data.shape[0]:  # the longest maturity does not have enough duration
        x = 0
        current_bond1, current_bond2 = aux_data['duration'].iloc[-2:].index
    else:
        a = aux_data['duration'].iloc[[dur_idx - 1, dur_idx]].values
        x = (dd - a[1]) / (a[0] - a[1])  # Ammount of bond 1
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

    # Loop for other dates
    paired_dates = zip(dates2loop[1:], dates2loop[:-1])
    for date, datem1 in tqdm(paired_dates, f'Backtesting NTNF {dd}y'):
        # get available bonds today
        aux_data = ntnf[ntnf['reference date'] == date].set_index('bond code')
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
            aux_data_select = aux_data[aux_data['du'] >= 21*rebalance_window]
            dur_idx = aux_data_select['duration'].searchsorted(dd)

            if dur_idx == 0:
                x = 1
                new_bond1, new_bond2 = aux_data_select['duration'].iloc[[0, 1]].index
            elif dur_idx == aux_data_select.shape[0]:  # the longest maturity does not have enough duration
                x = 0
                new_bond1, new_bond2 = aux_data_select['duration'].iloc[-2:].index
            else:
                a = aux_data_select['duration'].iloc[[dur_idx - 1, dur_idx]].values
                x = (dd - a[1]) / (a[0] - a[1])  # Ammount of bond 1
                new_bond1, new_bond2 = aux_data_select['duration'].iloc[[dur_idx - 1, dur_idx]].index

            df_bt.loc[date, 'bond 1'] = new_bond1
            df_bt.loc[date, 'bond 2'] = new_bond2
            df_bt.loc[date, 'du 1'] = aux_data.loc[new_bond1, 'du']
            df_bt.loc[date, 'du 2'] = aux_data.loc[new_bond2, 'du']

            sellvalue = df_bt.loc[datem1, 'quantity 1'] * (aux_data.loc[current_bond1, 'price'] - aux_data.loc[current_bond1, 'bidask spread'] / 2)
            sellvalue = sellvalue + df_bt.loc[datem1, 'quantity 2'] * (aux_data.loc[current_bond2, 'price'] - aux_data.loc[current_bond2, 'bidask spread'] / 2)
            sellvalue = sellvalue + df_bt.loc[datem1, 'quantity 1'] * aux_data.loc[current_bond1, 'coupon'] + df_bt.loc[datem1, 'quantity 2'] * aux_data.loc[current_bond2, 'coupon']

            df_bt.loc[date, 'quantity 1'] = x * sellvalue / (aux_data.loc[new_bond1, 'price'] + aux_data.loc[current_bond1, 'bidask spread'] / 2)
            df_bt.loc[date, 'quantity 2'] = (1 - x) * sellvalue / (aux_data.loc[new_bond2, 'price'] - aux_data.loc[current_bond2, 'bidask spread'] / 2)

            df_bt.loc[date, 'price 1'] = aux_data.loc[new_bond1, 'price']
            df_bt.loc[date, 'price 2'] = aux_data.loc[new_bond2, 'price']

            df_bt.loc[date, 'Notional'] = df_bt.loc[date, 'quantity 1'] * df_bt.loc[date, 'price 1'] \
                                        + df_bt.loc[date, 'quantity 2'] * df_bt.loc[date, 'price 2']

            current_bond1, current_bond2 = new_bond1, new_bond2

            next_rebalance_date = date + pd.DateOffset(months=rebalance_window)

    # Save the tracker composition
    filename = f'trackers/ntnf_{str(dd).replace(".", "p")}y.csv'
    df_bt.to_csv(DROPBOX.joinpath(filename), sep=';')

    # Standardize the tracker
    df_bt = 100 * df_bt['Notional'] / df_bt['Notional'].iloc[0]
    df_bt = df_bt.rename(f'NTNF {dd}y')
    df_tracker = pd.concat([df_tracker, df_bt], axis=1)

minutes = round((time() - tic) / 60, 1)
print('NTNF trackers took', minutes, 'minutes')
df_tracker.plot()
plt.show()

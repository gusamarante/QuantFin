"""
This routine computes the total return indexes for the brazilian federal bonds
"""

import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
import matplotlib.pyplot as plt
from quantfin.calendars import DayCounts
from quantfin.data import grab_connection, tracker_uploader

pd.options.display.max_columns = 50
pd.options.display.width = 200

start_date = '2006-01-01'
dc = DayCounts(dc='bus/252', calendar='anbima')

query = 'select * from raw_tesouro_nacional'
conn = grab_connection()
df_raw = pd.read_sql(query, con=conn)
df_raw['maturity_date'] = pd.to_datetime(df_raw['maturity_date'], dayfirst=True)
df_raw['reference_date'] = pd.to_datetime(df_raw['reference_date'], dayfirst=True)

df_raw = df_raw[df_raw['reference_date'] >= start_date]

all_trackers = pd.DataFrame()


# ===============
# ===== LFT =====
# ===============

# df = df_raw[df_raw['name'] == 'LFT']
# df_buy = df.pivot_table(index='reference_date', columns='maturity_date', values='max_price').dropna(how='all')
# df_sell = df.pivot_table(index='reference_date', columns='maturity_date', values='min_price').dropna(how='all')
#
# df_buy = df_buy.interpolate(limit_area='inside')
# df_sell = df_sell.interpolate(limit_area='inside')
#
# df_volume = df.pivot_table(index='reference_date', columns='maturity_date', values='volume').dropna(how='all')
# df_volume = df_volume.rolling(252).mean().shift(1).dropna(how='all')
#
# # ----- longest -----
# df_tracker = pd.DataFrame(columns=['Current', 'Ammount', 'Price', 'Notional'])
#
# current = df_buy.iloc[0].dropna().index.max()
# start_date = df_volume.index[0]
# df_tracker.loc[start_date, 'Current'] = current
# df_tracker.loc[start_date, 'Ammount'] = 1
# df_tracker.loc[start_date, 'Price'] = df_sell.loc[start_date, current]
# df_tracker.loc[start_date, 'Notional'] = df_tracker.loc[start_date, 'Price'] * df_tracker.loc[start_date, 'Ammount']
#
# date_loop = zip(df_volume.index[1:], df_volume.index[:-1])
#
# for date, datem1 in tqdm(date_loop, 'LFT - Mais Longa'):
#
#     new_current = df_buy.loc[date].dropna().index.max()
#
#     if new_current == current:  # Hold the position
#         df_tracker.loc[date, 'Current'] = df_tracker.loc[datem1, 'Current']
#         df_tracker.loc[date, 'Ammount'] = df_tracker.loc[datem1, 'Ammount']
#         df_tracker.loc[date, 'Price'] = df_sell.loc[date, current]
#         df_tracker.loc[date, 'Notional'] = df_tracker.loc[date, 'Price'] * df_tracker.loc[date, 'Ammount']
#
#     else:  # Roll the position
#         current = new_current
#
#         df_tracker.loc[date, 'Current'] = current
#         df_tracker.loc[date, 'Ammount'] = (df_tracker.loc[datem1, 'Ammount'] * df_sell.loc[date, current]) / df_buy.loc[
#             date, current]
#         df_tracker.loc[date, 'Price'] = df_sell.loc[date, current]
#         df_tracker.loc[date, 'Notional'] = df_tracker.loc[date, 'Price'] * df_tracker.loc[date, 'Ammount']
#
#     if np.abs(df_tracker.loc[date, 'Notional'] / df_tracker.loc[datem1, 'Notional'] - 1) >= 0.1:
#         df_tracker.loc[date, 'Ammount'] = df_tracker.loc[datem1, 'Ammount']
#         df_tracker.loc[date, 'Price'] = df_buy.loc[date, current]
#         df_tracker.loc[date, 'Notional'] = df_tracker.loc[date, 'Price'] * df_tracker.loc[date, 'Ammount']
#
#
# aux = (100 * df_tracker['Notional'] / df_tracker['Notional'].iloc[0]).rename('LFT Longa')
# all_trackers = pd.concat([all_trackers, aux], axis=1)
#
# # ----- most liquid -----
# df_tracker = pd.DataFrame(columns=['Current', 'Ammount', 'Price', 'Notional'])
#
# current = df_volume.iloc[0].idxmax()
# start_date = df_volume.index[0]
# df_tracker.loc[start_date, 'Current'] = current
# df_tracker.loc[start_date, 'Ammount'] = 1
# df_tracker.loc[start_date, 'Price'] = df_sell.loc[start_date, current]
# df_tracker.loc[start_date, 'Notional'] = df_tracker.loc[start_date, 'Price'] * df_tracker.loc[start_date, 'Ammount']
#
# date_loop = zip(df_volume.index[1:], df_volume.index[:-1])
#
# for date, datem1 in tqdm(date_loop, 'LFT - Mais Líquida'):
#
#     days2mat = dc.days(date, current)
#
#     if days2mat >= 1:  # Hold the position
#         df_tracker.loc[date, 'Current'] = df_tracker.loc[datem1, 'Current']
#         df_tracker.loc[date, 'Ammount'] = df_tracker.loc[datem1, 'Ammount']
#         df_tracker.loc[date, 'Price'] = df_sell.loc[date, current]
#         df_tracker.loc[date, 'Notional'] = df_tracker.loc[date, 'Price'] * df_tracker.loc[date, 'Ammount']
#
#     else:  # Roll the position
#         new_maturity = df_volume.loc[date].idxmax()
#         if new_maturity == current:
#             current = df_volume.loc[date].dropna().sort_values().index[-2]
#         else:
#             current = new_maturity
#
#         df_tracker.loc[date, 'Current'] = current
#         df_tracker.loc[date, 'Ammount'] = (df_tracker.loc[datem1, 'Ammount'] * df_sell.loc[date, current]) / df_buy.loc[
#             date, current]
#         df_tracker.loc[date, 'Price'] = df_sell.loc[date, current]
#         df_tracker.loc[date, 'Notional'] = df_tracker.loc[date, 'Price'] * df_tracker.loc[date, 'Ammount']
#
#     if np.abs(df_tracker.loc[date, 'Notional'] / df_tracker.loc[datem1, 'Notional'] - 1) >= 0.1:
#         df_tracker.loc[date, 'Ammount'] = df_tracker.loc[datem1, 'Ammount']
#         df_tracker.loc[date, 'Price'] = df_buy.loc[date, current]
#         df_tracker.loc[date, 'Notional'] = df_tracker.loc[date, 'Price'] * df_tracker.loc[date, 'Ammount']
#
# aux = (100 * df_tracker['Notional'] / df_tracker['Notional'].iloc[0]).rename('LFT Líquida')
# all_trackers = pd.concat([all_trackers, aux], axis=1)


# =================
# ===== NTN-B =====  se o ano de vencimento é impar, os cupons são maio e novembro, com vencimento em maio
# =================  se o ano de vencimento é par, os cupons são fevereiro e agosto, com vencimento em agosto
def get_coupon_dates(reference_date, maturity):
    ref_year = pd.to_datetime(reference_date).year
    mat_year = pd.to_datetime(maturity).year

    if mat_year % 2 == 0:  # Vencimento Par
        start_dt = pd.to_datetime(dt.date(ref_year, 2, 15))
        end_dt = pd.to_datetime(dt.date(mat_year, 8, 15))
        df_coupon_dates = pd.date_range(start=start_dt, end=end_dt, freq='12SMS')
    else:
        start_dt = pd.to_datetime(dt.date(ref_year, 5, 15))
        end_dt = pd.to_datetime(dt.date(mat_year, 5, 15))
        df_coupon_dates = pd.date_range(start=start_dt, end=end_dt, freq='12SMS')

    df_coupon_dates = dc.following(df_coupon_dates)

    return df_coupon_dates


df_vna = pd.read_excel('/Users/gustavoamarante/Dropbox/Personal Portfolio/VNA NTNB.xlsx', index_col=0)
df_vna = df_vna.resample('D').last().interpolate(limit_area='inside')
max_vna_date = max(df_vna.index)

df = df_raw[df_raw['name'] == 'NTN-B']

df_buy = df.pivot_table(index='reference_date', columns='maturity_date', values='max_price').dropna(how='all')
df_sell = df.pivot_table(index='reference_date', columns='maturity_date', values='min_price').dropna(how='all')

df_buy = df_buy.interpolate(limit_area='inside')
df_sell = df_sell.interpolate(limit_area='inside')

# ----- curta -----
df_tracker = pd.DataFrame(columns=['Current', 'Coupon', 'Ammount', 'Price', 'Notional'])

# Choose the first bond
current = df_buy.iloc[0].dropna().index.min()
start_date = df_buy.index[0]
roll_date = dc.following(pd.to_datetime(current) - pd.DateOffset(5))
coupon_dates = get_coupon_dates(df_buy.index[0], current)

df_tracker.loc[start_date, 'Current'] = current
df_tracker.loc[start_date, 'Coupon'] = 0
df_tracker.loc[start_date, 'Ammount'] = 1
df_tracker.loc[start_date, 'Price'] = df_sell.iloc[0][current]
df_tracker.loc[start_date, 'Notional'] = df_tracker.loc[start_date, 'Price'] * df_tracker.loc[start_date, 'Ammount']

date_loop = zip(df_buy.index[1:], df_buy.index[:-1])

for date, datem1 in tqdm(date_loop, 'NTNB - Curta'):
    if date == pd.to_datetime('2019-10-29'):
        a = 1

    if date > max_vna_date:
        break

    # check if there are coupon payments for this date
    if date in coupon_dates:
        df_tracker.loc[date, 'Coupon'] = df_vna.loc[date, 'VNA'] * (np.sqrt(1.06) - 1)
    else:
        df_tracker.loc[date, 'Coupon'] = 0

    days2end = (pd.to_datetime(df_buy[current].dropna().index[-1]) - date).days

    # Chooses the new bond to invest
    if (pd.to_datetime(date) >= roll_date) or (days2end <= 5):  # If it passed the roll date or if price will not be available
        new_current = sorted(df_buy.loc[date].dropna().index)[1]
        if new_current == current:
            new_current = sorted(df_buy.loc[date].dropna().index)[2]

        new_ammount_current = (df_tracker.loc[datem1, 'Ammount'] + df_tracker.loc[date, 'Coupon'] / df_buy.loc[date, current])
        df_tracker.loc[date, 'Ammount'] = new_ammount_current * df_sell.loc[date, current] / df_buy.loc[date, new_current]
        current = new_current
        roll_date = dc.following(pd.to_datetime(current) - pd.DateOffset(5))
        coupon_dates = get_coupon_dates(date, current)
    else:
        df_tracker.loc[date, 'Ammount'] = df_tracker.loc[datem1, 'Ammount'] + \
                                          df_tracker.loc[date, 'Coupon'] / df_buy.loc[date, current]

    df_tracker.loc[date, 'Current'] = current
    df_tracker.loc[date, 'Price'] = df_sell.loc[date, current]
    df_tracker.loc[date, 'Notional'] = df_tracker.loc[date, 'Price'] * df_tracker.loc[date, 'Ammount']

    if np.abs(df_tracker.loc[date, 'Notional'] / df_tracker.loc[datem1, 'Notional'] - 1) >= 0.5:
        df_tracker.loc[date, 'Ammount'] = df_tracker.loc[datem1, 'Ammount']
        df_tracker.loc[date, 'Price'] = df_sell.loc[datem1, current]
        df_tracker.loc[date, 'Notional'] = df_tracker.loc[date, 'Price'] * df_tracker.loc[date, 'Ammount']

df_tracker.to_clipboard()

aux = (100 * df_tracker['Notional'] / df_tracker['Notional'].iloc[0]).rename('NTNB CUrta')
all_trackers = pd.concat([all_trackers, aux], axis=1)

# aux = (100 * df_tracker['Notional'] / df_tracker['Notional'].iloc[0]).rename('NTNB Curta')
# all_trackers = pd.concat([all_trackers, aux], axis=1)
#
# # ----- longa -----
# df_tracker = pd.DataFrame(columns=['Current', 'Coupon', 'Ammount', 'Price', 'Notional'])
#
# # Choose the first bond
# current = df_buy.iloc[0].dropna().index.max()  # selects the current bond on the initial date
# coupon_dates = get_coupon_dates(df_buy.index[0], current)
#
# df_tracker.loc[df_buy.index[0], 'Current'] = current
# df_tracker.loc[df_buy.index[0], 'Coupon'] = 0
# df_tracker.loc[df_buy.index[0], 'Ammount'] = 1  # Number of bonds to buy
# df_tracker.loc[df_buy.index[0], 'Price'] = df_buy.iloc[0][current]
# df_tracker.loc[df_buy.index[0], 'Notional'] = df_tracker.loc[df_buy.index[0], 'Price'] * df_tracker.loc[
#     df_buy.index[0], 'Ammount']
#
# date_loop = zip(df_buy.index[1:], df_buy.index[:-1])
#
# for date, datem1 in tqdm(date_loop, 'NTNB - Longa'):
#
#     # check if there are coupon payments for this date
#     if date in coupon_dates:
#         df_tracker.loc[date, 'Coupon'] = df_vna.loc[pd.to_datetime(date), 'VNA'] * (np.sqrt(1.06) - 1)
#     else:
#         df_tracker.loc[date, 'Coupon'] = 0
#
#     # Chooses the new bond to invest
#     current = df_buy.loc[date].dropna().index.max()
#     if current == df_tracker.loc[datem1, 'Current']:
#         df_tracker.loc[date, 'Current'] = df_tracker.loc[datem1, 'Current']
#         df_tracker.loc[date, 'Ammount'] = df_tracker.loc[datem1, 'Ammount'] * (
#                     1 + df_tracker.loc[date, 'Coupon'] / df_buy.loc[date, current])
#
#     else:
#         df_tracker.loc[date, 'Current'] = current
#         df_tracker.loc[date, 'Ammount'] = df_tracker.loc[datem1, 'Ammount'] * (
#                     (df_sell.loc[date, current] + df_tracker.loc[date, 'Coupon']) / df_buy.loc[date, current])
#
#     df_tracker.loc[date, 'Price'] = df_sell.loc[date, current]
#     df_tracker.loc[date, 'Notional'] = df_tracker.loc[date, 'Price'] * df_tracker.loc[date, 'Ammount']
#
# aux = (100 * df_tracker['Notional'] / df_tracker['Notional'].iloc[0]).rename('NTNB Longa')
# all_trackers = pd.concat([all_trackers, aux], axis=1)
#
# all_trackers.index = pd.to_datetime(list(all_trackers.index))
# tracker_uploader(all_trackers)


# Plot
all_trackers.plot(legend='best')
plt.show()

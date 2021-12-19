import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
import matplotlib.pyplot as plt
from quantfin.calendars import DayCounts
from quantfin.data import grab_connection, tracker_uploader

query = 'select * from raw_tesouro_nacional'
conn = grab_connection()
df_raw = pd.read_sql(query, con=conn)

all_trackers = pd.DataFrame()

# ===============
# ===== LFT =====
# ===============
df = df_raw[df_raw['bond_type'] == 'LFT']
df_buy = df.pivot('reference_date', 'maturity', 'preco_compra').dropna(how='all')
df_sell = df.pivot('reference_date', 'maturity', 'preco_venda').dropna(how='all')

# ----- curto -----
df_tracker = pd.DataFrame(columns=['Current', 'Ammount', 'Price', 'Notional'])
df_tracker.loc[df_buy.index[0], 'Current'] = df_buy.iloc[0].dropna().index.min()
df_tracker.loc[df_buy.index[0], 'Ammount'] = 1
df_tracker.loc[df_buy.index[0], 'Price'] = df_buy.iloc[0][df_buy.iloc[0].dropna().index.min()]
df_tracker.loc[df_buy.index[0], 'Notional'] = df_tracker.loc[df_buy.index[0], 'Price'] * df_tracker.loc[
    df_buy.index[0], 'Ammount']

date_loop = zip(df_buy.index[1:], df_buy.index[:-1])

for date, datem1 in tqdm(date_loop, 'LFT - Curta'):

    current = df_buy.loc[date].dropna().index.min()
    if current == df_tracker.loc[datem1, 'Current']:
        df_tracker.loc[date, 'Current'] = df_tracker.loc[datem1, 'Current']
        df_tracker.loc[date, 'Ammount'] = df_tracker.loc[datem1, 'Ammount']
        df_tracker.loc[date, 'Price'] = df_sell.loc[date, current]
        df_tracker.loc[date, 'Notional'] = df_tracker.loc[date, 'Price'] * df_tracker.loc[date, 'Ammount']

    else:
        df_tracker.loc[date, 'Current'] = current
        df_tracker.loc[date, 'Ammount'] = df_tracker.loc[datem1, 'Notional'] / df_buy.loc[date, current]
        df_tracker.loc[date, 'Price'] = df_buy.loc[date, current]
        df_tracker.loc[date, 'Notional'] = df_tracker.loc[date, 'Price'] * df_tracker.loc[date, 'Ammount']

aux = (100 * df_tracker['Notional'] / df_tracker['Notional'].iloc[0]).rename('LFT Curta')
all_trackers = pd.concat([all_trackers, aux], axis=1)

# ----- longo -----
df_tracker = pd.DataFrame(columns=['Current', 'Ammount', 'Price', 'Notional'])
df_tracker.loc[df_buy.index[0], 'Current'] = df_buy.iloc[0].dropna().index.max()
df_tracker.loc[df_buy.index[0], 'Ammount'] = 1
df_tracker.loc[df_buy.index[0], 'Price'] = df_buy.iloc[0][df_buy.iloc[0].dropna().index.max()]
df_tracker.loc[df_buy.index[0], 'Notional'] = df_tracker.loc[df_buy.index[0], 'Price'] * df_tracker.loc[
    df_buy.index[0], 'Ammount']

date_loop = zip(df_buy.index[1:], df_buy.index[:-1])

for date, datem1 in tqdm(date_loop, 'LFT - Longa'):

    current = df_buy.loc[date].dropna().index.max()
    if current == df_tracker.loc[datem1, 'Current']:
        df_tracker.loc[date, 'Current'] = df_tracker.loc[datem1, 'Current']
        df_tracker.loc[date, 'Ammount'] = df_tracker.loc[datem1, 'Ammount']
        df_tracker.loc[date, 'Price'] = df_sell.loc[date, current]
        df_tracker.loc[date, 'Notional'] = df_tracker.loc[date, 'Price'] * df_tracker.loc[date, 'Ammount']

    else:
        df_tracker.loc[date, 'Current'] = current
        df_tracker.loc[date, 'Ammount'] = df_tracker.loc[datem1, 'Notional'] / df_buy.loc[date, current]
        df_tracker.loc[date, 'Price'] = df_buy.loc[date, current]
        df_tracker.loc[date, 'Notional'] = df_tracker.loc[date, 'Price'] * df_tracker.loc[date, 'Ammount']

aux = (100 * df_tracker['Notional'] / df_tracker['Notional'].iloc[0]).rename('LFT Longa')
all_trackers = pd.concat([all_trackers, aux], axis=1)


# =================
# ===== NTN-B =====  se o ano de vencimento é impar, os cupons são maio e novembro, com vencimento em maio
# =================  se o ano de vencimento é par, os cupons são fevereiro e agosto, com vencimento em agosto
def get_coupon_dates(reference_date, maturity):
    ref_year = pd.to_datetime(reference_date).year
    mat_year = pd.to_datetime(maturity).year

    if mat_year % 2 == 0:  # Vencimento Par
        start_date = pd.to_datetime(dt.date(ref_year, 2, 15))
        end_date = pd.to_datetime(dt.date(mat_year, 8, 15))
        df_coupon_dates = pd.date_range(start=start_date, end=end_date, freq='12SMS')
    else:
        start_date = pd.to_datetime(dt.date(ref_year, 5, 15))
        end_date = pd.to_datetime(dt.date(mat_year, 5, 15))
        df_coupon_dates = pd.date_range(start=start_date, end=end_date, freq='12SMS')

    df_coupon_dates = dc.preceding(df_coupon_dates - pd.DateOffset(1))

    return df_coupon_dates


df_vna = pd.read_excel('/Users/gustavoamarante/Dropbox/Personal Portfolio/VNA NTNB.xlsx', index_col=0)
dc = DayCounts('bus/252', calendar='anbima')
df = df_raw[df_raw['bond_type'] == 'NTNB']
df_buy = df.pivot_table(index='reference_date', columns='maturity', values='preco_compra', aggfunc='mean').dropna(how='all')
df_sell = df.pivot_table(index='reference_date', columns='maturity', values='preco_venda', aggfunc='mean').dropna(how='all')


# ----- curta -----
df_tracker = pd.DataFrame(columns=['Current', 'Coupon', 'Ammount', 'Price', 'Notional'])

# Choose the first bond
current = df_buy.iloc[0].dropna().index.min()  # selects the current bond on the initial date
roll_date = dc.following(pd.to_datetime(current) - pd.DateOffset(5))
coupon_dates = get_coupon_dates(df_buy.index[0], current)

df_tracker.loc[df_buy.index[0], 'Current'] = current
df_tracker.loc[df_buy.index[0], 'Coupon'] = 0
df_tracker.loc[df_buy.index[0], 'Ammount'] = 1  # Number of bonds to buy
df_tracker.loc[df_buy.index[0], 'Price'] = df_buy.iloc[0][current]
df_tracker.loc[df_buy.index[0], 'Notional'] = df_tracker.loc[df_buy.index[0], 'Price'] * df_tracker.loc[
    df_buy.index[0], 'Ammount']

date_loop = zip(df_buy.index[1:], df_buy.index[:-1])

for date, datem1 in tqdm(date_loop, 'NTNB - Curta'):

    # check if there are coupon payments for this date
    if date in coupon_dates:
        df_tracker.loc[date, 'Coupon'] = df_vna.loc[pd.to_datetime(date), 'VNA'] * (np.sqrt(1.06) - 1)
    else:
        df_tracker.loc[date, 'Coupon'] = 0

    days2end = (pd.to_datetime(df_buy[current].dropna().index[-1]) - pd.to_datetime(date)).days

    # Chooses the new bond to invest
    if (pd.to_datetime(date) >= roll_date) or (days2end <= 5):
        new_current = sorted(df_buy.loc[date].dropna().index)[1]
        df_tracker.loc[date, 'Ammount'] = (df_tracker.loc[datem1, 'Ammount'] * (df_sell.loc[date, current] + df_tracker.loc[date, 'Coupon'])/df_buy.loc[date, new_current])
        current = new_current
        roll_date = dc.following(pd.to_datetime(current) - pd.DateOffset(5))
        coupon_dates = get_coupon_dates(date, current)
    else:
        df_tracker.loc[date, 'Ammount'] = df_tracker.loc[datem1, 'Ammount'] * (1 + df_tracker.loc[date, 'Coupon']/df_buy.loc[date, current])

    df_tracker.loc[date, 'Current'] = current
    df_tracker.loc[date, 'Price'] = df_sell.loc[date, current]
    df_tracker.loc[date, 'Notional'] = df_tracker.loc[date, 'Price'] * df_tracker.loc[date, 'Ammount']

aux = (100 * df_tracker['Notional'] / df_tracker['Notional'].iloc[0]).rename('NTNB Curta')
all_trackers = pd.concat([all_trackers, aux], axis=1)


# ----- longa -----
df_tracker = pd.DataFrame(columns=['Current', 'Coupon', 'Ammount', 'Price', 'Notional'])

# Choose the first bond
current = df_buy.iloc[0].dropna().index.max()  # selects the current bond on the initial date
coupon_dates = get_coupon_dates(df_buy.index[0], current)

df_tracker.loc[df_buy.index[0], 'Current'] = current
df_tracker.loc[df_buy.index[0], 'Coupon'] = 0
df_tracker.loc[df_buy.index[0], 'Ammount'] = 1  # Number of bonds to buy
df_tracker.loc[df_buy.index[0], 'Price'] = df_buy.iloc[0][current]
df_tracker.loc[df_buy.index[0], 'Notional'] = df_tracker.loc[df_buy.index[0], 'Price'] * df_tracker.loc[
    df_buy.index[0], 'Ammount']

date_loop = zip(df_buy.index[1:], df_buy.index[:-1])

for date, datem1 in tqdm(date_loop, 'NTNB - Longa'):

    # check if there are coupon payments for this date
    if date in coupon_dates:
        df_tracker.loc[date, 'Coupon'] = df_vna.loc[pd.to_datetime(date), 'VNA'] * (np.sqrt(1.06) - 1)
    else:
        df_tracker.loc[date, 'Coupon'] = 0

    # Chooses the new bond to invest
    current = df_buy.loc[date].dropna().index.max()
    if current == df_tracker.loc[datem1, 'Current']:
        df_tracker.loc[date, 'Current'] = df_tracker.loc[datem1, 'Current']
        df_tracker.loc[date, 'Ammount'] = df_tracker.loc[datem1, 'Ammount'] * (1 + df_tracker.loc[date, 'Coupon']/df_buy.loc[date, current])

    else:
        df_tracker.loc[date, 'Current'] = current
        df_tracker.loc[date, 'Ammount'] = df_tracker.loc[datem1, 'Ammount'] * ((df_sell.loc[date, current] + df_tracker.loc[date, 'Coupon']) / df_buy.loc[date, current])

    df_tracker.loc[date, 'Price'] = df_sell.loc[date, current]
    df_tracker.loc[date, 'Notional'] = df_tracker.loc[date, 'Price'] * df_tracker.loc[date, 'Ammount']

aux = (100 * df_tracker['Notional'] / df_tracker['Notional'].iloc[0]).rename('NTNB Longa')
all_trackers = pd.concat([all_trackers, aux], axis=1)

all_trackers.index = pd.to_datetime(list(all_trackers.index))
tracker_uploader(all_trackers)

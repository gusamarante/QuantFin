import pandas as pd
from tqdm import tqdm
from quantfin.data import grab_connection
import matplotlib.pyplot as plt

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

aux = (100 * df_tracker['Notional'] / df_tracker['Notional'].iloc[0]).rename('LFT Curto')
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

aux = (100 * df_tracker['Notional'] / df_tracker['Notional'].iloc[0]).rename('LFT Longo')
all_trackers = pd.concat([all_trackers, aux], axis=1)


# =================
# ===== NTN-B =====
# =================





all_trackers.plot()
plt.show()
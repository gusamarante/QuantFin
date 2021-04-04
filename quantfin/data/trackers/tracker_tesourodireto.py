import pandas as pd
from tqdm import tqdm

df_raw = pd.read_excel('/Users/gustavoamarante/Dropbox/Personal Portfolio/tesouro_direto.xlsx',
                       index_col=0)

all_trackers = pd.DataFrame()

# TODO 'Tesouro IPCA+ com Juros Semestrais', 'Tesouro IPCA+',
#      'Tesouro Prefixado com Juros Semestrais', 'Tesouro Prefixado',
#      'Tesouro IGPM+ com Juros Semestrais', 'Tesouro Selic'

# ===== Tesouro Selic =====
df = df_raw[df_raw['Tipo Titulo'] == 'Tesouro Selic']
df_buy = df.pivot('Data Base', 'Data Vencimento', 'PU Compra Manha')
df_sell = df.pivot('Data Base', 'Data Vencimento', 'PU Venda Manha')

# ----- curto -----
df_tracker = pd.DataFrame(columns=['Current', 'Ammount', 'Price', 'Notional'])
df_tracker.loc[df_buy.index[0], 'Current'] = df_buy.iloc[0].dropna().index.min()
df_tracker.loc[df_buy.index[0], 'Ammount'] = 1
df_tracker.loc[df_buy.index[0], 'Price'] = df_buy.iloc[0][df_buy.iloc[0].dropna().index.min()]
df_tracker.loc[df_buy.index[0], 'Notional'] = df_tracker.loc[df_buy.index[0], 'Price'] * df_tracker.loc[df_buy.index[0], 'Ammount']

date_loop = zip(df_buy.index[1:], df_buy.index[:-1])

for date, datem1 in tqdm(date_loop, 'Tesouro Selic - Curta'):

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

aux = (100 * df_tracker['Notional'] / df_tracker['Notional'].iloc[0]).rename('Tesouro Selic Curto')
all_trackers = pd.concat([all_trackers, aux], axis=1)

# ----- longo -----
df_tracker = pd.DataFrame(columns=['Current', 'Ammount', 'Price', 'Notional'])
df_tracker.loc[df_buy.index[0], 'Current'] = df_buy.iloc[0].dropna().index.max()
df_tracker.loc[df_buy.index[0], 'Ammount'] = 1
df_tracker.loc[df_buy.index[0], 'Price'] = df_buy.iloc[0][df_buy.iloc[0].dropna().index.max()]
df_tracker.loc[df_buy.index[0], 'Notional'] = df_tracker.loc[df_buy.index[0], 'Price'] * df_tracker.loc[df_buy.index[0], 'Ammount']

date_loop = zip(df_buy.index[1:], df_buy.index[:-1])

for date, datem1 in tqdm(date_loop, 'Tesouro Selic - Longa'):

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

aux = (100 * df_tracker['Notional'] / df_tracker['Notional'].iloc[0]).rename('Tesouro Selic Longo')
all_trackers = pd.concat([all_trackers, aux], axis=1)

# ===== Save to Excel =====
# TODO this should go to a SQL database
writer = pd.ExcelWriter('/Users/gustavoamarante/Dropbox/Personal Portfolio/trackers_tesouro_direto.xlsx')
all_trackers.to_excel(writer)
writer.save()

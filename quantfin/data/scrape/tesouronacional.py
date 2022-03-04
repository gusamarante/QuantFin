"""
Scrapper of Brazilian bond prices on the secondary market.
Compiled by the brazilian central bank.
"""

import pandas as pd
from tqdm import tqdm
from quantfin.data import grab_connection
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

start_date = '2003-01-01'

rename_dict = {'DATA MOV': 'reference_date',
               'SIGLA': 'name',
               'CODIGO': 'cb_code',
               'CODIGO ISIN': 'isin_code',
               'EMISSAO': 'issuance_date',
               'VENCIMENTO': 'maturity_date',
               'NUM DE OPER': 'number_of_trades',
               'QUANT NEGOCIADA': 'volume',
               'VALOR NEGOCIADO': 'financial_volume',
               'PU MIN': 'min_price',
               'PU MED': 'mid_price',
               'PU MAX': 'max_price',
               'PU LASTRO': 'backing_price',
               'VALOR PAR': 'face_value',
               'TAXA MIN': 'min_rate',
               'TAXA MED': 'mid_rate',
               'TAXA MAX': 'max_rate',
               'NUM OPER COM CORRETAGEM': 'number_of_trades_brokers',
               'QUANT NEG COM CORRETAGEM': 'volume_brokers'}


def scrape_url(datestr):
    return f'https://www4.bcb.gov.br/pom/demab/negociacoes/download/NegT{datestr}.ZIP'


dates2scrape = pd.date_range(start='2003-01-01',
                             end=pd.to_datetime('today'),
                             freq='M')

df_raw = pd.DataFrame(columns=rename_dict.values())

for date in tqdm(dates2scrape):
    url = scrape_url(date.strftime('%Y%m'))
    df_aux = pd.read_csv(url, sep=';', decimal=",")
    df_aux = df_aux.rename(rename_dict, axis=1)
    df_aux['maturity_date'] = pd.to_datetime(df_aux['maturity_date'], dayfirst=True)
    df_aux['issuance_date'] = pd.to_datetime(df_aux['issuance_date'], dayfirst=True)
    df_aux['reference_date'] = pd.to_datetime(df_aux['reference_date'], dayfirst=True)
    df_raw = pd.concat([df_raw, df_aux], axis=0)

# Save to database
conn = grab_connection()
df_raw.to_sql('raw_tesouro_nacional', con=conn, if_exists='replace', index=False)

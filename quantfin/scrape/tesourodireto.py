"""
This routine grabs the prices of the brazilian bonds that are available on the "tesouro direto" plataform and puts
the data on the sqlite file.
"""

import pandas as pd
from quantfin.data import grab_connection

url = r'https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv'

df = pd.read_csv(url, sep=';', decimal=',')
df['Data Vencimento'] = pd.to_datetime(df['Data Vencimento'], dayfirst=True)
df['Data Base'] = pd.to_datetime(df['Data Base'], dayfirst=True)

rename_dict = {'Tipo Titulo': 'bond_type',
               'Data Vencimento': 'maturity',
               'Data Base': 'reference_date',
               'Taxa Compra Manha': 'taxa_compra',
               'Taxa Venda Manha': 'taxa_venda',
               'PU Compra Manha': 'preco_compra',
               'PU Venda Manha': 'preco_venda',
               'PU Base Manha': 'preco_base'}

df = df.rename(rename_dict, axis=1)

# upload data
conn = grab_connection()
df.to_sql('raw_tesouro_direto', con=conn, if_exists='replace', index=False)

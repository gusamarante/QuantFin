"""
This routine grabs the prices of the brazilian bonds that are available on the "tesouro direto" plataform
"""

import pandas as pd

url = r'https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv'

df = pd.read_csv(url, sep=';', decimal=',')
df['Data Vencimento'] = pd.to_datetime(df['Data Vencimento'], dayfirst=True)
df['Data Base'] = pd.to_datetime(df['Data Base'], dayfirst=True)

# TODO this should go to a SQL database
writer = pd.ExcelWriter('/Users/gustavoamarante/Dropbox/Personal Portfolio/raw_tesouro_direto.xlsx')
df.to_excel(writer)
writer.save()

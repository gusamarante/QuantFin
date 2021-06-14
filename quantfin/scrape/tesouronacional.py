"""
This routine scrapes prices for the brazilian government bonds.
This routine works for all bonds.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from quantfin.data import grab_connection

url = r'https://sisweb.tesouro.gov.br/apex/f?p=2031:2:0::::'

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

download_tags = soup.find_all('a', {"style": "padding-right:5px;"})

download_links = []
for tag in download_tags:
    download_links.append(r'https://sisweb.tesouro.gov.br/apex/' + tag.attrs['href'])

# loop on everything
df = pd.DataFrame(columns=['Taxa Compra Manhã', 'Taxa Venda Manhã', 'PU Compra Manhã', 'PU Venda Manhã',
                           'PU Base Manhã', 'bond_name', 'maturity'])

for link in tqdm(download_links, 'Looping every link'):
    xls = pd.ExcelFile(link)

    for name in xls.sheet_names:

        # grab the bond name
        bond_name = name.split()[0].replace('-', '')
        if name.split()[1] == 'Princ':
            bond_name = bond_name + 'P'

        # grab the bond maturity
        maturity = pd.to_datetime(name.split()[-1], dayfirst=True)

        # read the data
        df_aux = pd.read_excel(xls, name, skiprows=1)
        df_aux.columns = df_aux.columns.str.replace('9:00', 'Manhã')
        df_aux = df_aux.rename({'PU Extrato Manhã': 'PU Base Manhã'}, axis=1)
        df_aux = df_aux.dropna(how='all', axis=1)

        try:
            df_aux['Dia'] = pd.to_datetime(df_aux['Dia'], dayfirst=True)
        except ValueError:
            print(f'Deu ruim na {bond_name} que vence em {maturity}')
            continue

        df_aux['bond_name'] = bond_name
        df_aux['maturity'] = maturity

        # concatenate
        df = pd.concat([df, df_aux], axis=0, ignore_index=True)

# rename variables
rename_dict = {'bond_name': 'bond_type',
               'maturity': 'maturity',
               'Dia': 'reference_date',
               'Taxa Compra Manhã': 'taxa_compra',
               'Taxa Venda Manhã': 'taxa_venda',
               'PU Compra Manhã': 'preco_compra',
               'PU Venda Manhã': 'preco_venda',
               'PU Base Manhã': 'preco_base'}

df = df.rename(rename_dict, axis=1)

# Save to database
conn = grab_connection()
df.to_sql('raw_tesouro_nacional', con=conn, if_exists='replace', index=False)

"""
This routine scrapes prices for the brazilian government bonds.
This routine works for all bonds. Not only the ones available for 'tesouro direto'.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

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

        df_aux['bond_name'] = bond_name
        df_aux['maturity'] = maturity

        # concatenate
        df = pd.concat([df, df_aux], axis=0, ignore_index=True)

# TODO this should go to a SQL database
writer = pd.ExcelWriter('/Users/gustavoamarante/Dropbox/Personal Portfolio/raw_titulos_publicos.xlsx')
df.to_excel(writer)
writer.save()

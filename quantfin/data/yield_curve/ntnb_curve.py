from quantfin.data import DROPBOX
from tqdm import tqdm
from time import time
import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)


# Read the Data
last_year = 2022
raw_data = pd.DataFrame()

for year in tqdm(range(2003, last_year + 1), 'Reading files'):
    aux = pd.read_csv(DROPBOX.joinpath(f'trackers/dados_ntnb {year}.csv'), sep=';')
    raw_data = pd.concat([raw_data, aux])

raw_data = raw_data.drop(['Unnamed: 0', 'index'], axis=1)
raw_data['reference date'] = pd.to_datetime(raw_data['reference date'])


# Function to generate ntnb cashflows
def ntnb_cashflows(reference_date, maturity_date, vna):
    mat_year = maturity_date.year

    # Se o ano é par ou impar, define o end date
    # Start date vai ser algo muito velho
    # filtra para datas maiores que a reference
    # adiciona os valores dos coupons

    return


# EXAMPLE - Single date
today = raw_data['reference date'].max()
current_bonds = raw_data[raw_data['reference date'] == today].sort_values('du')





a = 1

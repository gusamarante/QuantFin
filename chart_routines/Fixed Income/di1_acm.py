"""
Generate the estimate of term premium of the DI1 based on the ACM Model
"""

import getpass
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from quantfin.data import DROPBOX
from scipy.stats import percentileofscore
from time import time

tic = time()

# TODO remove this
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 250)

username = getpass.getuser()
save_path = Path(f'/Users/{username}/Dropbox/Aulas/Insper - Renda Fixa/2023')

# Read the DIs
df_di = pd.DataFrame()
for year in tqdm(range(2006, 2023 + 1), 'Reading Files'):
    aux = pd.read_csv(DROPBOX.joinpath(f'trackers/dados_di1 {year}.csv'),
                      sep=';')
    df_di = pd.concat([df_di, aux], axis=0)
df_di = df_di.drop(['Unnamed: 0'], axis=1)
df_di['reference_date'] = pd.to_datetime(df_di['reference_date'])
df_di['maturity_date'] = pd.to_datetime(df_di['maturity_date'])

# build the curve
df_curve = df_di.pivot('reference_date', 'du', 'rate')
df_curve = df_curve.interpolate(axis=1, method='cubic')
df_curve = df_curve.dropna(how='any', axis=1)
df_curve.index = pd.to_datetime(df_curve.index)

# Keep the relevant maturities
mat2keep = [63, 126] + [252*x for x in range(1, 10)]
df_curve = df_curve[mat2keep]

# Grab Total Return Index for the DI
df_tr = pd.read_csv(DROPBOX.joinpath(f'trackers/trackers di1.csv'), sep=';', index_col=0)
df_tr.index = pd.to_datetime(df_tr.index)

print(round((time() - tic)/60, 1), 'minutes')

print(df_tr)

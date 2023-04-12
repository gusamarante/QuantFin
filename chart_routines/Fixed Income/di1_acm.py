"""
Generate the estimate of term premium of the DI1 based on the ACM Model
"""

import getpass
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from quantfin.data import DROPBOX, SGS
from scipy.stats import percentileofscore
from time import time
from quantfin.models import NominalACM

tic = time()

# TODO remove this
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 250)

start_date = '2007-01-01'
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


# ===== Custom Functions =====
def get_excess_returns(df, funding):

    excess_returns = pd.DataFrame(columns=df.columns)
    funding = (np.log(1 + funding)).resample('M').last()

    for n in df.columns[1:]:
        p_n_minus_1_d = np.exp(-df[n - 1] * ((n - 1) / 12))
        p_n_d_minus_1 = (np.exp(-df[n] * (n / 12))).shift(1)
        ret_n_d = np.log(p_n_minus_1_d / p_n_d_minus_1)
        excess_returns[n] = ret_n_d - funding * (1 / 12)

    return excess_returns


# ===== build the curve =====
# Grab CDI
sgs = SGS()
df_cdi = sgs.fetch({12: 'CDI'})
df_cdi = (1+df_cdi['CDI']/100)**252 - 1

# pivot observed points data
df_curve = df_di.pivot(index='reference_date', columns='du', values='rate')

# fill the 0 date with the CDI
df_curve[0] = df_cdi

# Flat-Forward Interpolation
df_curve = np.log(1/((1 + df_curve)**(df_curve.columns/252)))
df_curve = df_curve.interpolate(axis=1, method='linear', limit_area='inside')
df_curve = (1 / np.exp(df_curve)) ** (252 / df_curve.columns) - 1
df_curve[0] = df_cdi

df_curve = df_curve.dropna(how='any', axis=1)
df_curve.index = pd.to_datetime(df_curve.index)
mat2keep = [21*x for x in range(0, int(df_curve.columns.max() / 21) + 1)]
df_curve = df_curve[mat2keep]
df_curve.columns = (df_curve.columns/21).astype(int)
df_curve = df_curve.resample('M').last()

# Get excess returns
df_exp_curve = np.log(1 + df_curve)
df_ret = get_excess_returns(df_exp_curve, df_cdi)
df_ret = df_ret.dropna(how='all', axis=0).dropna(how='all', axis=1)

df_curve = df_curve[df_curve.index >= start_date]
df_ret = df_ret[df_ret.index >= start_date]

# =================
# ===== Model =====
# =================

acm = NominalACM(curve=df_curve, excess_returns=df_ret, compute_miy=True, verbose=True, freq='monthly')

for mat in [12, 24, 36, 60]:
    df_plot = pd.concat([acm.curve[mat].rename('Yield'),
                         acm.miy[mat].rename('Model Implied'),
                         acm.term_premium[mat].rename('Term Premium')],
                        axis=1)

    df_plot.plot(title=f'{mat/12}-year')
    plt.tight_layout()
    plt.show()

print(round((time() - tic)/60, 1), 'minutes')

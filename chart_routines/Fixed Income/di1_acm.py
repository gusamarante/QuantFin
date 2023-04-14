"""
Generate the estimate of term premium of the DI1 based on the ACM Model

Next Steps:
- Make this run in daily frequency
- Implied central bank trajectory on both the observed and the risk neutral yield
"""

import getpass
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from quantfin.models import NominalACM
from quantfin.data import DROPBOX, SGS

tic = time()

# TODO remove this
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 250)

size = 7
start_date = '2007-01-01'
username = getpass.getuser()
save_path = Path(f'/Users/{username}/Dropbox/Aulas/Insper - Renda Fixa/2023/figures')

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
    fig = plt.figure(figsize=(size * (16 / 9), size))


    df_plot = pd.concat([acm.curve[mat].rename('Observed'),
                         acm.miy[mat].rename('Model Implied'),
                         acm.rny[mat].rename('Risk-Neutral')],
                        axis=1)
    df_plot = 100 * df_plot

    ax_yields = plt.subplot2grid((1, 2), (0, 0))
    ax_yields.plot(df_plot)
    ax_yields.set_title(f'{int(mat/12)}-year Yields')
    ax_yields.set_ylim((2, 18))
    ax_yields.grid(axis='y', alpha=0.3)
    ax_yields.grid(axis='x', alpha=0.3)
    ax_yields.legend(df_plot.columns, frameon=True, loc='best')
    locators = mdates.YearLocator()
    ax_yields.xaxis.set_major_locator(locators)
    ax_yields.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_yields.tick_params(labelrotation=90, axis='x')

    ax_tp = plt.subplot2grid((1, 2), (0, 1))
    ax_tp.plot(acm.term_premium[mat].rename('Term Premium')*100)
    ax_tp.axhline(0, color='black', linewidth=0.5)
    ax_tp.set_title(f'{int(mat/12)}-year Term Premium')
    ax_tp.set_ylim((-2, 7))
    ax_tp.grid(axis='y', alpha=0.3)
    ax_tp.grid(axis='x', alpha=0.3)
    locators = mdates.YearLocator()
    ax_tp.xaxis.set_major_locator(locators)
    ax_tp.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_tp.tick_params(labelrotation=90, axis='x')

    plt.tight_layout()
    plt.savefig(save_path.joinpath(f'DI1 ACM Term Premium {int(mat/12)}y.pdf'))
    plt.show()

print(round((time() - tic)/60, 1), 'minutes')

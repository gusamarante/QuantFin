import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from quantfin.statistics import marchenko_pastur, detone, cov2corr
from quantfin.portfolio import Markowitz, BlackLitterman, HRP

# User defined parameters
ew_com = 21 * 3

# fixed parameters
file_path = r'/Users/gustavoamarante/Dropbox/CQF/Final Project/'  # Mac
# file_path = r'/Users/gusamarante/Dropbox/CQF/Final Project/'  # Macbook

tic = time()
# =========================
# ===== READ THE DATA =====
# =========================
# Read Bloomberg Tickers and create dictionaries for renaming
df_tickers = pd.read_excel(file_path + r'Data - BBG Data Values.xlsx',
                           index_col=0, sheet_name='Tickers')

tr_dict = df_tickers['Total Return Index (UBS)'].to_dict()
tr_dict = {v: k for k, v in tr_dict.items()}

fwd_dict = df_tickers['Forward 3m (in bps)'].to_dict()
fwd_dict = {v: k for k, v in fwd_dict.items()}

spot_dict = df_tickers['Spot'].to_dict()
spot_dict = {v: k for k, v in spot_dict.items()}

ppp_dict = df_tickers['PPP'].to_dict()
ppp_dict = {v: k for k, v in ppp_dict.items()}

# Read Total Return Index
df_tr = pd.read_excel(file_path + r'Data - BBG Data Values.xlsx',
                      index_col=0, sheet_name='Total Return')
df_tr = df_tr.rename(tr_dict, axis=1)

# Read fwds
df_fwd = pd.read_excel(file_path + r'Data - BBG Data Values.xlsx',
                       index_col=0, sheet_name='FWD 3M')
df_fwd = df_fwd.rename(fwd_dict, axis=1)

# Read Spot
df_spot = pd.read_excel(file_path + r'Data - BBG Data Values.xlsx',
                        index_col=0, sheet_name='Spot')
df_spot = df_spot.rename(spot_dict, axis=1)

# Read PPP
df_ppp = pd.read_excel(file_path + r'Data - BBG Data Values.xlsx',
                       index_col=0, sheet_name='PPP')
df_ppp = df_ppp.rename(ppp_dict, axis=1)

# Read Libor
df_libor = pd.read_excel(file_path + r'Data - LIBOR.xlsx',
                         index_col=0, sheet_name='LIBOR', na_values=['#N/A'])
df_libor = df_libor.fillna(method='ffill') / 100


# ========================
# ===== COMPUTATIONS =====
# ========================
# exponentially weighted covariance/correlation
df_cov = df_tr.pct_change(1).ewm(ew_com, min_periods=ew_com).cov()
df_cov = df_cov.dropna(how='all') * 3 * 21  # covariances from daily to quarterly

df_corr = df_tr.pct_change(1).ewm(ew_com, min_periods=ew_com).corr()
df_corr = df_corr.dropna(how='all')

# Carry
df_carry = (df_spot + df_fwd/10000) / df_spot - 1

# Value
df_value = (1 - df_ppp / df_spot) * 0.0445
df_value = df_value.dropna(axis=1, how='all')

# Momentum
df_mom = df_tr.pct_change(21 * 3)

# ============================
# ===== STATIC PORTFOLIO =====
# ============================

# date for the static portfolio
last_date = df_libor.index[-1]

# 3-month volatility
vols = pd.Series(data=np.sqrt(df_cov.loc[last_date].values.diagonal()),
                 index=df_tr.columns, name='Vol')

# Dataframe that is going to hold the weights of different methods for comparison.
df_weights = pd.DataFrame(index=df_tr.columns)

# ----- equal weighted -----
df_weights['Equal Weighted'] = 1 / df_tr.shape[1]

# ----- inverse volatility -----
aux = 1 / vols
df_weights['Inverse Volatility'] = aux / aux.sum()

# ----- Hierarchical Risk Parity -----
hrp = HRP(cov=df_cov.loc[last_date])
df_weights['Hierarchical Risk Parity'] = hrp.weights

# ----- Detoned HRP -----
corr_detoned = detone(df_corr.loc[last_date])
dhrp = HRP(cov=df_cov.loc[last_date], corr=corr_detoned)
df_weights['Detoned Hierarchical Risk Parity'] = dhrp.weights

# ----- Black-Litterman + Max Sharpe -----
# generate the matrices of views
P = pd.DataFrame()
v = pd.Series()

# add carry views
for ccy in df_tr.columns:
    try:
        v.loc[f'{ccy} carry'] = df_carry.loc[last_date, ccy]
        P.loc[f'{ccy} carry', ccy] = 1
    except KeyError:  # If a currency does not have a carry signal, skips this view.
        continue

# add value views
for ccy in df_tr.columns:
    try:
        v.loc[f'{ccy} value'] = df_value.loc[last_date, ccy]
        P.loc[f'{ccy} value', ccy] = 1
    except KeyError:  # If a currency does not have a value signal, skips this view.
        continue


P = P.fillna(0)
v = v.to_frame('Views')

# denoise the covariance
mp_corr, _, _ = marchenko_pastur(df_corr.loc[last_date],
                                 T=21 * 3, N=df_tr.shape[1])

mp_cov = pd.DataFrame(data=np.diag(vols) @ mp_corr @ np.diag(vols),
                      index=vols.index, columns=vols.index)

bl = BlackLitterman(sigma=mp_cov,
                    estimation_error=1 / (21 * 3),
                    views_p=P,
                    views_v=v,
                    w_equilibrium=df_weights['Inverse Volatility'].to_frame(),
                    avg_risk_aversion=1.2,
                    mu_historical=df_mom.loc[last_date].to_frame('Historical'),
                    mu_shrink=0.99,  # needs to be tuned
                    overall_confidence=100)  # needs to be tuned

vol_bl = pd.Series(data=np.sqrt(np.diag(bl.sigma_bl)), index=bl.sigma_bl.index)
corr_bl = cov2corr(bl.sigma_bl)

mkw = Markowitz(mu=bl.mu_bl,
                sigma=vol_bl,
                corr=corr_bl,
                rf=(1 + df_libor.loc[last_date, 'US 3m LIBOR']) ** 0.25 - 1,
                risk_aversion=1.2)

df_weights['Marchanko-Pastur + Black-Litterman'] = mkw.risky_weights


# === END ===
toc = time()
print(round(toc - tic, 1), 'seconds')

# ===== CHART =====
df_plot = df_weights.sort_index(ascending=False)
ax = df_plot.plot(kind='barh', figsize=(6, 10), width=0.8)
plt.grid(axis='x')
plt.axvline(0, color='black', linewidth=1)
plt.tight_layout()
plt.savefig(file_path + r'figures/Static Weights.pdf', pad_inches=0)
plt.show()


import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from quantfin.statistics import marchenko_pastur, detone, cov2corr
from quantfin.portfolio import Markowitz, BlackLitterman, HRP, Performance

# User defined parameters
mu_shrink = 0.99
overall_confidence = 100

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
# total returns
df_returns = df_tr.pct_change(1)

# exponentially weighted covariance/correlation
df_cov = df_returns.ewm(ew_com, min_periods=ew_com).cov()
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

# ==========================
# ===== EQUAL WEIGHTED =====
# ==========================
df_etf = pd.DataFrame()
aux_ret = df_returns.mean(axis=1).dropna()
aux_ret = (1 + aux_ret).cumprod()
aux_ret = 100 * aux_ret / aux_ret.iloc[0]
df_etf['Equal Weighted'] = aux_ret

# ==============================================
# ===== MARCHENKO-PASTUR + BLACK LITTERMAN =====
# ==============================================
# starting date of the backtest
initial_date_carry, end_date_carry = df_carry.dropna(how='all').index[[0, -1]]
initial_date_value, end_date_value = df_value.dropna(how='all').index[[0, -1]]
initial_date_mom, end_date_mom = df_mom.dropna(how='all').index[[0, -1]]
initial_date_tr, end_date_tr = df_tr.dropna(how='all').index[[0, -1]]
initial_date_libor, end_date_libor = df_libor.dropna(how='all').index[[0, -1]]

start_date = max(initial_date_carry, initial_date_value, initial_date_mom, initial_date_tr, initial_date_libor)
end_date = min(end_date_carry, end_date_value, end_date_mom, end_date_tr, end_date_libor)

cond1 = df_tr.index >= start_date
cond2 = df_tr.index <= end_date
calendar = list(df_tr.index[cond1 & cond2])
next_rebalance_date = start_date

# Empty DataFrames
df_vols = pd.DataFrame(columns=df_tr.columns)
weights_mpbl = pd.DataFrame(columns=df_tr.columns)
weights_iv = pd.DataFrame(columns=df_tr.columns)
weights_hrp = pd.DataFrame(columns=df_tr.columns)
weights_dhrp = pd.DataFrame(columns=df_tr.columns)

for date in tqdm(calendar, 'Marchenk-Pastur + Black-Litterman'):
    # computations
    vols = pd.Series(data=np.sqrt(df_cov.loc[date].values.diagonal()), index=df_tr.columns, name='Vol').dropna()
    df_vols.loc[date] = vols
    weights_iv.loc[date] = (1 / vols) / ((1 / vols).sum())

    if date >= next_rebalance_date:

        # Generate the views
        P = pd.DataFrame(columns=df_tr.columns)
        v = pd.Series(dtype=float)

        # add carry views
        carry_countries = df_carry.loc[date].dropna().index
        for ccy in carry_countries:
            try:
                v.loc[f'{ccy} carry'] = df_carry.loc[date, ccy]
                P.loc[f'{ccy} carry', ccy] = 1
            except KeyError:  # If a currency does not have a carry signal, skips this view.
                continue

        # add value views
        value_countries = df_value.loc[date].dropna().index
        for ccy in value_countries:
            try:
                v.loc[f'{ccy} value'] = df_value.loc[date, ccy]
                P.loc[f'{ccy} value', ccy] = 1
            except KeyError:  # If a currency does not have a value signal, skips this view.
                continue

        P = P.fillna(0)
        v = v.to_frame('Views')

        # denoise the covariance matrix
        mp_corr, _, _ = marchenko_pastur(df_corr.loc[date], T=21 * 3, N=vols.shape[0])

        mp_cov = pd.DataFrame(data=np.diag(vols) @ mp_corr @ np.diag(vols),
                              index=vols.index, columns=vols.index)

        # optimization
        bl = BlackLitterman(sigma=mp_cov,
                            estimation_error=1 / (21 * 3),
                            views_p=P,
                            views_v=v,
                            w_equilibrium=weights_iv.loc[date].to_frame(),
                            avg_risk_aversion=1.2,
                            mu_historical=df_mom.loc[date].to_frame('Historical'),
                            mu_shrink=mu_shrink,  # needs to be tuned
                            overall_confidence=overall_confidence)  # needs to be tuned

        vol_bl = pd.Series(data=np.sqrt(np.diag(bl.sigma_bl)), index=bl.sigma_bl.index)
        corr_bl = cov2corr(bl.sigma_bl)

        mkw = Markowitz(mu=bl.mu_bl,
                        sigma=vol_bl,
                        corr=corr_bl,
                        rf=(1 + df_libor.loc[date, 'US 3m LIBOR']) ** 0.25 - 1,
                        risk_aversion=1.2)

        weights_mpbl.loc[date] = mkw.risky_weights
        next_rebalance_date = date + pd.DateOffset(months=1)


next_rebalance_date = start_date
for date in tqdm(calendar, 'HRP'):
    if date >= next_rebalance_date:
        try:
            hrp = HRP(cov=df_cov.loc[date])
            corr_detoned = detone(df_corr.loc[date])
            dhrp = HRP(cov=df_cov.loc[date], corr=corr_detoned)
        except ValueError:
            continue

        weights_hrp.loc[date] = hrp.weights
        weights_dhrp.loc[date] = dhrp.weights
        next_rebalance_date = date + pd.DateOffset(months=1)

# Compute the total return indexes
# inverse vol
weights_iv = weights_iv.resample('M').last().reindex(df_tr.index).fillna(method='ffill')
aux_ret = (weights_iv * df_returns).dropna().sum(axis=1)
aux_ret = (1 + aux_ret).cumprod()
aux_ret = 100 * aux_ret / aux_ret.iloc[0]
df_etf['Inverse Volatility'] = aux_ret

# marchenko pastur + black litterman
weights_mpbl = weights_mpbl.reindex(df_tr.index)
weights_mpbl = weights_mpbl.fillna(method='ffill')
aux_ret = (weights_mpbl * df_returns).dropna().sum(axis=1)
aux_ret = (1 + aux_ret).cumprod()
aux_ret = 100 * aux_ret / aux_ret.iloc[0]
df_etf['Marchenko-Pastur and Black-Litterman'] = aux_ret

# HRP
weights_hrp = weights_hrp.resample('M').last().reindex(df_tr.index)
weights_hrp = weights_hrp.fillna(method='ffill')
aux_ret = (weights_hrp * df_returns).dropna().sum(axis=1)
aux_ret = (1 + aux_ret).cumprod()
aux_ret = 100 * aux_ret / aux_ret.iloc[0]
df_etf['Hierarchical Risk Parity'] = aux_ret

# detoned HRP
weights_dhrp = weights_dhrp.resample('M').last().reindex(df_tr.index)
weights_dhrp = weights_dhrp.fillna(method='ffill')
aux_ret = (weights_dhrp * df_returns).dropna().sum(axis=1)
aux_ret = (1 + aux_ret).cumprod()
aux_ret = 100 * aux_ret / aux_ret.iloc[0]
df_etf['Detoned Hierarchical Risk Parity'] = aux_ret

# tide up
df_etf = df_etf.dropna()
df_etf = 100 * df_etf / df_etf.iloc[0]

# print performance
perf = Performance(df_etf)
print(perf.sharpe)

# Saves to excel
writer = pd.ExcelWriter(file_path + r'output - ETFization.xlsx')
df_etf.to_excel(writer, 'ETFs')
perf.table.to_excel(writer, 'Perf')
writer.save()



# Chart
df_etf.plot(figsize=(11, 6))
plt.show()

from quantfin.portfolio import Performance, MaxSharpe
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)

# ===== Read the data =====
# file_path = "/Users/gustavoamarante/Dropbox/Personal Portfolio/trackers/"  # iMac
file_path = r"C:/Users/gamarante/Dropbox/Personal Portfolio/trackers/"

# tracker
df_trackers = pd.read_csv(file_path + 'fx_trackers.csv', index_col='date')
df_trackers.index = pd.to_datetime(df_trackers.index)

# Spot
df_spot = pd.read_csv(file_path + 'fx_spot.csv', index_col='date')
df_spot = df_spot.drop(['SEK.1', 'ZAR.1', 'ZAR.2', 'ZAR.3', 'ZAR.4', 'TRY.1', 'CNH.1'], axis=1)
df_spot.index = pd.to_datetime(df_spot.index)
df_spot = df_spot.reindex(df_trackers.index).fillna(method='ffill')

# ppp
df_ppp = pd.read_csv(file_path + 'fx_ppp.csv', index_col='date')
df_ppp = df_ppp.drop(['SEK.1', 'ZAR.1', 'ZAR.2', 'ZAR.3', 'ZAR.4', 'TRY.1', 'CNH.1'], axis=1)
df_ppp.index = pd.to_datetime(df_ppp.index)

df_ppp = df_ppp.resample('D').last().fillna(method='ffill')
df_ppp = df_ppp.reindex(df_trackers.index).fillna(method='ffill')

# carry
df_carry3m = pd.read_csv(file_path + 'fx_carry3m.csv', index_col='date')
df_carry3m = df_carry3m.drop(['THB.1', 'THB.2', 'TRY.1', 'CNH.1'], axis=1)
df_carry3m.index = pd.to_datetime(df_carry3m.index)

df_carry12m = pd.read_csv(file_path + 'fx_carry12m.csv', index_col='date')
df_carry12m = df_carry12m.drop(['THB.1', 'THB.2', 'TRY.1', 'CNH.1'], axis=1)
df_carry12m.index = pd.to_datetime(df_carry12m.index)

# ===== Estimate of Covariance =====
df_cov = df_trackers.pct_change().resample('D').last().ewm(com=21*3).cov() * 252
df_vol = df_trackers.pct_change().ewm(com=21*3).std() * np.sqrt(252)

# =================
# ===== Value =====
# =================

# build value signal
df_value = -(df_spot.divide(df_ppp) - 1)
df_value = df_value.dropna(how='all')

# Absolute weights
w_abs = (np.sign(df_value) * df_vol).dropna(how='all', axis=1)
w_abs = w_abs.ewm(com=60).mean()  # Weight Smoothing
w_abs = w_abs.shift(1)  # MOST IMPORTANT LINE

w_abs = w_abs.resample('M').last().reindex(df_trackers.index).fillna(method='ffill')

# Relative Weights
w_rel = df_value.ewm(com=60).mean().rank(axis=1).subtract(df_value.ewm(com=60).mean().rank(axis=1).mean(axis=1), axis=0)
w_rel = w_rel.divide(w_rel.abs().sum(axis=1)/2, axis=0)
w_rel = w_rel.shift(1)  # MOST IMPORTANT LINE

w_rel = w_rel.resample('M').last().reindex(df_trackers.index).fillna(method='ffill')

# Maximum Exposure
w_max = pd.DataFrame(columns=w_rel.columns, index=df_value.index)
next_rebal_date = df_value.index[1]

for date in tqdm(df_value.index, 'Maximum exposure weights'):

    if date >= next_rebal_date:
        mu = df_value.shift(1).loc[date].dropna()  # MOST IMPORTANT LINE
        cov = df_cov.shift(30).loc[date].loc[mu.index, mu.index]  # MOST IMPORTANT LINE

        try:
            port = MaxSharpe(mu, cov, 0, gross_risk=0.3)
            w_max.loc[date] = port.risky_weights
        except RuntimeError:
            w_max.loc[date] = np.nan

        next_rebal_date = next_rebal_date + pd.DateOffset(months=1)

    else:
        continue

w_max = w_max.reindex(df_trackers.index).fillna(method='ffill')

# ===== Backtest Value =====
# Absolute Weights
bt_abs_value = (df_trackers.pct_change() * w_abs).dropna(how='all').dropna(how='all', axis=1)
bt_abs_value = bt_abs_value.sum(axis=1)
bt_abs_value = (1 + bt_abs_value).cumprod()
bt_abs_value = 100 * bt_abs_value / bt_abs_value.iloc[0]
bt_abs_value = bt_abs_value.rename('Absolute Value')

# Relative Weights
bt_rel_value = (df_trackers.pct_change() * w_rel).dropna(how='all').dropna(how='all', axis=1)
bt_rel_value = bt_rel_value.sum(axis=1)
bt_rel_value = (1 + bt_rel_value).cumprod()
bt_rel_value = 100 * bt_rel_value / bt_rel_value.iloc[0]
bt_rel_value = bt_rel_value.rename('Relative Value')

# Maximum Exposure Weights
bt_max_value = (df_trackers.pct_change() * w_max).dropna(how='all').dropna(how='all', axis=1)
bt_max_value = bt_max_value.sum(axis=1)
bt_max_value = (1 + bt_max_value).cumprod()
bt_max_value = 100 * bt_max_value / bt_max_value.iloc[0]
bt_max_value = bt_max_value.rename('Maximum Exposure Value')


# =================
# ===== Carry =====
# =================

# build carry signal
df_carry = df_carry3m
df_carry = df_carry.dropna(how='all')

# Absolute weights
w_abs = (np.sign(df_carry) * df_vol).dropna(how='all', axis=1)
w_abs = w_abs.ewm(com=60).mean()  # Weight Smoothing
w_abs = w_abs.shift(1)  # MOST IMPORTANT LINE

w_abs = w_abs.resample('M').last().reindex(df_trackers.index).fillna(method='ffill')

# Relative Weights
w_rel = df_carry.ewm(com=60).mean().rank(axis=1).subtract(df_carry.ewm(com=60).mean().rank(axis=1).mean(axis=1), axis=0)
w_rel = w_rel.divide(w_rel.abs().sum(axis=1)/2, axis=0)
w_rel = w_rel.shift(1)  # MOST IMPORTANT LINE

w_rel = w_rel.resample('M').last().reindex(df_trackers.index).fillna(method='ffill')

# Maximum Exposure
# w_max = pd.DataFrame(columns=w_rel.columns, index=df_carry.index)
# next_rebal_date = df_carry.index[1]
#
# for date in tqdm(df_carry.index, 'Maximum exposure weights'):
#
#     if date >= next_rebal_date:
#         mu = df_carry.shift(1).loc[date].dropna()  # MOST IMPORTANT LINE
#         cov = df_cov.shift(30).loc[date].loc[mu.index, mu.index]  # MOST IMPORTANT LINE
#
#         try:
#             port = MaxSharpe(mu, cov, 0, gross_risk=0.5)
#             w_max.loc[date] = port.risky_weights
#         except RuntimeError:
#             w_max.loc[date] = np.nan
#
#         next_rebal_date = next_rebal_date + pd.DateOffset(months=1)
#
#     else:
#         continue
#
# w_max = w_max.reindex(df_trackers.index).fillna(method='ffill')

# ===== Backtest Carry =====
# Absolute Weights
bt_abs_carry = (df_trackers.pct_change() * w_abs).dropna(how='all').dropna(how='all', axis=1)
bt_abs_carry = bt_abs_carry.sum(axis=1)
bt_abs_carry = (1 + bt_abs_carry).cumprod()
bt_abs_carry = 100 * bt_abs_carry / bt_abs_carry.iloc[0]
bt_abs_carry = bt_abs_carry.rename('Absolute Carry')

# Relative Weights
bt_rel_carry = (df_trackers.pct_change() * w_rel).dropna(how='all').dropna(how='all', axis=1)
bt_rel_carry = bt_rel_carry.sum(axis=1)
bt_rel_carry = (1 + bt_rel_carry).cumprod()
bt_rel_carry = 100 * bt_rel_carry / bt_rel_carry.iloc[0]
bt_rel_carry = bt_rel_carry.rename('Relative Carry')

# Maximum Exposure Weights
# bt_max_carry = (df_trackers.pct_change() * w_max).dropna(how='all').dropna(how='all', axis=1)
# bt_max_carry = bt_max_carry.sum(axis=1)
# bt_max_carry = (1 + bt_max_carry).cumprod()
# bt_max_carry = 100 * bt_max_carry / bt_max_carry.iloc[0]
# bt_max_carry = bt_max_carry.rename('Maximum Exposure Carry')


# ====================
# ===== Momentum =====
# ====================

# build value signal
df_mom = df_trackers.pct_change(252)
df_mom = df_mom.dropna(how='all')

# Absolute weights
w_abs = (np.sign(df_mom) * df_vol).dropna(how='all', axis=1)
w_abs = w_abs.ewm(com=60).mean()  # Weight Smoothing
w_abs = w_abs.shift(1)  # MOST IMPORTANT LINE

w_abs = w_abs.resample('M').last().reindex(df_trackers.index).fillna(method='ffill')

# Relative Weights
w_rel = df_mom.ewm(com=60).mean().rank(axis=1).subtract(df_value.ewm(com=60).mean().rank(axis=1).mean(axis=1), axis=0)
w_rel = w_rel.divide(w_rel.abs().sum(axis=1)/2, axis=0)
w_rel = w_rel.shift(1)  # MOST IMPORTANT LINE

w_rel = w_rel.resample('M').last().reindex(df_trackers.index).fillna(method='ffill')

# Maximum Exposure
w_max = pd.DataFrame(columns=w_rel.columns, index=df_mom.index)
next_rebal_date = df_mom.index[1]

for date in tqdm(df_mom.index, 'Maximum exposure weights'):

    if date >= next_rebal_date:
        mu = df_mom.shift(1).loc[date].dropna()  # MOST IMPORTANT LINE
        cov = df_cov.shift(30).loc[date].loc[mu.index, mu.index]  # MOST IMPORTANT LINE

        try:
            port = MaxSharpe(mu, cov, 0, gross_risk=0.3)
            w_max.loc[date] = port.risky_weights
        except RuntimeError:
            w_max.loc[date] = np.nan

        next_rebal_date = next_rebal_date + pd.DateOffset(months=1)

    else:
        continue

w_max = w_max.reindex(df_trackers.index).fillna(method='ffill')

# ===== Backtest Value =====
# Absolute Weights
bt_abs_mom = (df_trackers.pct_change() * w_abs).dropna(how='all').dropna(how='all', axis=1)
bt_abs_mom = bt_abs_mom.sum(axis=1)
bt_abs_mom = (1 + bt_abs_mom).cumprod()
bt_abs_mom = 100 * bt_abs_mom / bt_abs_mom.iloc[0]
bt_abs_mom = bt_abs_mom.rename('Absolute Momentum')

# Relative Weights
bt_rel_mom = (df_trackers.pct_change() * w_rel).dropna(how='all').dropna(how='all', axis=1)
bt_rel_mom = bt_rel_mom.sum(axis=1)
bt_rel_mom = (1 + bt_rel_mom).cumprod()
bt_rel_mom = 100 * bt_rel_mom / bt_rel_mom.iloc[0]
bt_rel_mom = bt_rel_mom.rename('Relative Momentum')

# Maximum Exposure Weights
bt_max_mom = (df_trackers.pct_change() * w_max).dropna(how='all').dropna(how='all', axis=1)
bt_max_mom = bt_max_mom.sum(axis=1)
bt_max_mom = (1 + bt_max_mom).cumprod()
bt_max_mom = 100 * bt_max_mom / bt_max_mom.iloc[0]
bt_max_mom = bt_max_mom.rename('Maximum Exposure Momentum')

# Concatenate and compare
df_bt = pd.concat([bt_abs_value, bt_rel_value, bt_max_value,
                   bt_abs_carry, bt_rel_carry,
                   bt_abs_mom, bt_rel_mom, bt_max_mom], axis=1)

perf = Performance(df_bt)
print(perf.table)

print(df_bt.pct_change().corr())

df_bt.plot()
plt.show()

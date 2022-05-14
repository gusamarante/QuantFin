import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

file_path = r'/Users/gustavoamarante/Dropbox/Aulas/Insper - Financas Quantitativas/2022/Monitoria 1/fx_data.xlsx'
df_trackers = pd.read_excel(file_path, index_col=0, sheet_name='trackers')
df_carry = pd.read_excel(file_path, index_col=0, sheet_name='carry')
df_value = pd.read_excel(file_path, index_col=0, sheet_name='ppp_value')

# Resample to Monthly Frequancy
df_trackers = df_trackers.resample('M').last()
df_carry = df_carry.resample('M').last()
df_value = df_value.resample('M').last()

# Currencies to keep
commom_ccy = (df_trackers.columns.intersection(df_carry.columns)).intersection(df_value.columns)
commom_ccy = list(set(commom_ccy) - {'ARS', 'TRY'})
df_trackers = df_trackers[commom_ccy]
df_carry = df_carry[commom_ccy]
df_value = df_value[commom_ccy]
df_mom = df_trackers.pct_change(12)  # we could tune this


# Custom Functions
def rank_weights(df_signal):
    df_signal = df_signal.dropna(how='all')
    ranks = df_signal.rank(axis=1)
    avg_ranks = ranks.subtract(ranks.mean(axis=1), axis=0)
    sum_ranks = avg_ranks.abs().sum(axis=1)
    weights = 2 * avg_ranks.div(sum_ranks, axis=0)
    return weights


# Build the Dollar Factor Portfolio
fac_usd = df_trackers.pct_change(1).mean(axis=1).fillna(0)
fac_usd = 100 * (1 + fac_usd).cumprod()
fac_usd = fac_usd.rename('Dollar')

# Build the Momentum Factor
weights_mom = rank_weights(df_mom)
fac_mom = (weights_mom.shift(1) * df_trackers.pct_change(1)).dropna(how='all').sum(axis=1)
fac_mom = 100 * (1 + fac_mom).cumprod()
fac_mom = fac_mom.rename('Momentum')

# Build the Carry Factor
weights_carry = rank_weights(df_carry)
fac_carry = (weights_carry.shift(1) * df_trackers.pct_change(1)).dropna(how='all').sum(axis=1)
fac_carry = 100 * (1 + fac_carry).cumprod()
fac_carry = fac_carry.rename('Carry')

# Build the Value Factor
weights_value = rank_weights(df_value)
fac_value = (weights_value.shift(1) * df_trackers.pct_change(1)).dropna(how='all').sum(axis=1)
fac_value = 100 * (1 + fac_value).cumprod()
fac_value = fac_value.rename('Value')

# put factors together
df_factors = pd.concat([fac_usd, fac_mom, fac_carry, fac_value], axis=1)
print(df_factors.pct_change(1).corr())

# Pooled regression
pooled_data = pd.DataFrame(columns=['Return', 'Momentum', 'Carry', 'Value'])

for ccy in df_trackers.columns:
    ccy_block = pd.concat([df_trackers[ccy].pct_change(1).dropna().rename('Return'),
                           df_mom[ccy].shift(1).rename('Momentum'),
                           df_carry[ccy].shift(1).rename('Carry'),
                           df_value[ccy].shift(1).rename('Value')],
                          axis=1)
    ccy_block = ccy_block.dropna()
    pooled_data = pd.concat([pooled_data, ccy_block])

X_pooled = sm.add_constant(pooled_data[['Momentum', 'Carry', 'Value']])
Y_pooled = pooled_data['Return']
mod = sm.OLS(Y_pooled.astype(float), X_pooled.astype(float), missing='drop')
res = mod.fit()
print(res.summary())

# Betas to the factors
X = sm.add_constant(df_factors.pct_change(1).dropna())
df_betas = pd.DataFrame(columns=X.columns)

for ccy in df_trackers.columns:
    Y = df_trackers[ccy].pct_change(1).reindex(X.index)
    mod = sm.OLS(Y, X, missing='drop')
    res = mod.fit()
    df_betas.loc[ccy] = res.params

print(df_betas)

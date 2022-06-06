import matplotlib.pyplot as plt
from quantfin.data import SGS
import pandas as pd
import numpy as np

sgs = SGS()
df_cdi = sgs.fetch({12: 'CDI'})
df_cdi = df_cdi['CDI'] / 100
df_cdi = (1 + df_cdi) ** 21 - 1
df_cdi.index = pd.to_datetime(df_cdi.index)

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)

df_trackers = pd.read_csv(r'/Users/gustavoamarante/Dropbox/Personal Portfolio/trackers/trackers_DI1.csv',
                          sep=';', index_col=0)
df_trackers.index = pd.to_datetime(df_trackers.index)

df_raw = pd.read_csv(r'/Users/gustavoamarante/Dropbox/Personal Portfolio/trackers/Dados DI1.csv',
                     sep=';', index_col=0)
df_raw['reference_date'] = pd.to_datetime(df_raw['reference_date'])
df_raw['maturity_date'] = pd.to_datetime(df_raw['maturity_date'])

df_raw.loc[119290, 'pnl'] = -76.86 * (254 / 296)  # Correcting a wrong value

days = df_trackers.count()
ret = (df_trackers.iloc[-1] / 100) ** (252/days) - 1
vols = df_trackers.pct_change(1).std() * np.sqrt(252)
sharpe = ret / vols

print(df_trackers.pct_change(1).corr())
print(df_trackers.resample('M').last().pct_change(1).corr())
print(df_trackers.resample('Y').last().pct_change(1).corr())

ret.plot(kind='bar', title='Annualized Returns')
plt.tight_layout()
plt.show()

vols.plot(kind='bar', title='Annualized Volatility')
plt.tight_layout()
plt.show()

plt.plot(vols, ret, '-o')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.tight_layout()
plt.show()

sharpe.plot(kind='bar', title='Annualized Sharpe Ratio')
plt.tight_layout()
plt.show()

# Vol-weighted strategy
weights = 1 / (df_trackers.pct_change(1).resample('M').std() * np.sqrt(252)).shift(1)
weights = weights.reindex(df_trackers.index).fillna(method='bfill')
weights = weights.div(weights.sum(axis=1), axis=0)
volw_returns = (df_trackers.pct_change(1) * weights).sum(axis=1)

volw_tracker = 100 * (1 + volw_returns).cumprod()


# Carry/vol - mesmo dv01
df_curve = df_raw.pivot(index='reference_date', columns='du', values='rate')
df_curve = df_curve.interpolate(axis=1, limit_area='inside', method='cubic')
df_curve = df_curve.dropna(axis=1)

df_carry = pd.DataFrame()
for mat in np.arange(0.5, 10.5, 0.5):
    try:
        carry = ((1 + df_curve[252 * mat]) ** mat) / (((1 + df_curve[252 * mat - 21]) ** (mat - 21/252)) * (1 + df_cdi)) - 1
    except KeyError:
        continue

    carry = carry.dropna()
    carry = carry.rename(f'{mat} years')
    df_carry = pd.concat([df_carry, carry], axis=1)

df_carry = (1 + df_carry) ** 12 - 1
df_carry.index = pd.to_datetime(df_carry.index)

df_carry.plot()
plt.tight_layout()
plt.show()

# ===== Backtest Carry =====

shorts = df_carry.resample('M').last().idxmin(axis=1)
longs = df_carry.resample('M').last().idxmax(axis=1)

shorts = shorts.reindex(df_trackers.index).fillna(method='ffill')
longs = longs.reindex(df_trackers.index).fillna(method='ffill')

for date in df_trackers.index:
    a = 1

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)

df_trackers = pd.read_csv(r'/Users/gustavoamarante/Dropbox/Personal Portfolio/trackers/trackers_DI1.csv',
                          sep=';', index_col=0)
df_trackers.index = pd.to_datetime(df_trackers.index)

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

sharpe.plot(kind='bar', title='Annualized Sharpe Ratio')
plt.tight_layout()
plt.show()

plt.plot(vols, ret, '-o')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.tight_layout()
plt.show()

# Vol-weighted strategy
weights = 1 / (df_trackers.pct_change(1).resample('M').std() * np.sqrt(252)).shift(1)
weights = weights.reindex(df_trackers.index).fillna(method='bfill')
weights = weights.div(weights.sum(axis=1), axis=0)
volw_returns = (df_trackers.pct_change(1) * weights).sum(axis=1)

volw_tracker = 100 * (1 + volw_returns).cumprod()


# Carry/vol - mesmo dv01

a = 1

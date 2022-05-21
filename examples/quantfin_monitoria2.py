import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Change the address according to your computer
file = r'/Users/gustavoamarante/Dropbox/Aulas/Insper - Financas Quantitativas/2022/Monitoria 2/Dados Monitoria 2.xlsx'
df = pd.read_excel(file, sheet_name='Sheet1', index_col=0)
df = df.drop(['LFT Curta', 'LFT Longa'], axis=1)

# Ex ante vol
vols = df.pct_change(1).ewm(com=60).std() * np.sqrt(252)
vols = vols.shift(1)
vols = vols.resample('M').last()

returns = df.resample('M').last().pct_change(1)

retvol = returns / vols

# Predictability - Equation 2 / Fig 1
for s in retvol.columns:
    tstat = pd.Series(name='t-stat', dtype=float)

    for h in range(1, 13):
        aux = pd.concat([retvol[s].shift(h).rename('X'), retvol[s].rename('Y')], axis=1)
        results = smf.ols('Y ~ X', data=aux).fit()
        tstat.loc[h] = results.tvalues['X']

    tstat.plot(kind='bar', title=s)
    plt.axhline(0, lw=1, color='black')
    plt.axhline(2, lw=1, color='red', ls='--')
    plt.axhline(-2, lw=1, color='red', ls='--')
    plt.tight_layout()
    plt.show()


# Predictability - Equation 3 / Fig 1
for s in retvol.columns:
    tstat = pd.Series(name='t-stat', dtype=float)

    for h in range(1, 13):
        aux = pd.concat([np.sign(retvol[s].shift(h).rename('X')), retvol[s].rename('Y')], axis=1)
        results = smf.ols('Y ~ X', data=aux).fit()
        tstat.loc[h] = results.tvalues['X']

    tstat.plot(kind='bar', title=s)
    plt.axhline(0, lw=1, color='black')
    plt.axhline(2, lw=1, color='red', ls='--')
    plt.axhline(-2, lw=1, color='red', ls='--')
    plt.tight_layout()
    plt.show()

# Timeseries Momentum Factor - Eq 5 and 5.1 / Fig 2
ret12m = df.resample('M').last().pct_change(12)
ret1m = df.resample('M').last().pct_change(1)
signals = np.sign(ret12m.shift(1))

tsmom_assets = signals * (0.4 / vols.shift(1)) * ret1m
tsmom_assets_eri = 100 * (1 + tsmom_assets).cumprod()

tsmom = tsmom_assets.mean(axis=1)
tsmom_eri = 100 * (1 + tsmom.dropna()).cumprod()

tsmom_eri = pd.concat([tsmom_assets_eri, tsmom_eri.rename('TSM Factor')], axis=1)

n_months = tsmom_eri.count()
ann_ret = (tsmom_eri.iloc[-1] / tsmom_eri.fillna(method='bfill').iloc[0]) ** (12 / n_months) - 1
ann_vol = tsmom_eri.pct_change(1).std() * np.sqrt(12)
sharpe = (ann_ret / ann_vol).sort_values(ascending=False)

sharpe.plot(kind='bar')
plt.tight_layout()
plt.show()

# Passive Long VS TSM Factor - Fig 3

# TSMoM smile - Fig 4

# EXTRA - Cross-Section Momentum Factor



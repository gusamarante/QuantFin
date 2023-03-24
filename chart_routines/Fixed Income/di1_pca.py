import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.decomposition import PCA
from quantfin.data import DROPBOX
from pathlib import Path
import getpass

# User defined parameters
username = getpass.getuser()
save_path = Path(f'/Users/{username}/Dropbox/Aulas/Insper - Renda Fixa/2023/figures')

# get data
df_raw = pd.DataFrame()
for year in tqdm(range(2006, 2023 + 1), 'Reading Files'):
    aux = pd.read_csv(DROPBOX.joinpath(f'trackers/dados_di1 {year}.csv'),
                      sep=';')
    df_raw = pd.concat([df_raw, aux], axis=0)

df_raw = df_raw.drop(['Unnamed: 0'], axis=1)
df_raw['reference_date'] = pd.to_datetime(df_raw['reference_date'])
df_raw['maturity_date'] = pd.to_datetime(df_raw['maturity_date'])

# build the curve
df_curve = df_raw.pivot(index='reference_date', columns='du', values='rate')
df_curve = df_curve.interpolate(axis=1, method='cubic', limit_area='inside')
df_curve = df_curve.dropna(how='any', axis=1)
df_curve.index = pd.to_datetime(df_curve.index)


# ===========================
# ===== Full sample PCA =====
# ===========================
pca = PCA(n_components=3)
pca.fit(df_curve.values)

df_var_full = pd.DataFrame(data=pca.explained_variance_ratio_)
df_loadings_full = pd.DataFrame(data=pca.components_.T,
                                columns=['PC 1', 'PC 2', 'PC 3'],
                                index=df_curve.columns)
df_mean_full = pd.DataFrame(data=pca.mean_, index=df_curve.columns,
                            columns=['Médias'])
df_pca_full = pd.DataFrame(data=pca.transform(df_curve.values),
                           columns=['PC 1', 'PC 2', 'PC 3'],
                           index=df_curve.index)

signal = np.sign(df_loadings_full.iloc[-1])
df_loadings_full = df_loadings_full * signal
df_pca_full = df_pca_full * signal


# ==================
# ===== Charts =====
# ==================
# Curve dynamics
df_plot = df_curve[[252*a for a in range(1, 10)]] * 100

plt.figure(figsize=(7, 7 * (9 / 16)))
plt.plot(df_plot)

last_date = df_curve.index[-1]
plt.title(f"Yields of the DI Curve as of {last_date.strftime('%d/%b/%y')}")


ax = plt.gca()
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.tick_params(rotation=90, axis='x')
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.legend(df_plot.columns, frameon=True, loc='best', ncol=2)

plt.tight_layout()
plt.savefig(save_path.joinpath('DI Yields.pdf'))
plt.show()
plt.close()


# Time Series of the PCs
df_plot = df_pca_full

plt.figure(figsize=(7, 7 * (9 / 16)))
plt.plot(df_plot)


last_date = df_curve.index[-1]
plt.title(f"Principal Components of the DI Curve as of {last_date.strftime('%d/%b/%y')}")


ax = plt.gca()
ax.axhline(0, color='black', linewidth=0.5)
locators = mdates.YearLocator()
ax.xaxis.set_major_locator(locators)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.tick_params(rotation=90, axis='x')
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.legend(df_plot.columns, frameon=True, loc='best')

plt.tight_layout()
plt.savefig(save_path.joinpath('DI Principal Components.pdf'))
plt.show()
plt.close()

# Factor Loadings
df_plot = df_loadings_full * 100

plt.figure(figsize=(7, 7 * (9 / 16)))
plt.plot(df_plot)
plt.xlabel('Maturity (DU)')

last_date = df_curve.index[-1]
plt.title(f"Factor Loadings of the DI Curve as of {last_date.strftime('%d/%b/%y')}")


ax = plt.gca()
ax.axhline(0, color='black', linewidth=0.5)
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.legend(df_plot.columns, frameon=True, loc='best')

plt.tight_layout()
plt.savefig(save_path.joinpath('DI Factor Loadings.pdf'))
plt.show()
plt.close()

# Explained Variance
df_plot = df_var_full

plt.figure(figsize=(7, 7 * (9 / 16)))
plt.bar([f'PC{a}' for a in range(1, 4)], df_plot.iloc[:, 0].values)
plt.plot(df_plot.iloc[:, 0].cumsum(), color='orange')

last_date = df_curve.index[-1]
plt.title(f"Explained Variance of the DI Curve as of {last_date.strftime('%d/%b/%y')}")


ax = plt.gca()
ax.axhline(0, color='black', linewidth=0.5)
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig(save_path.joinpath('DI Explained Variance.pdf'))
plt.show()
plt.close()




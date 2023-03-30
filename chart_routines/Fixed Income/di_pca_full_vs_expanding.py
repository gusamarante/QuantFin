"""
genarates the expanding window PCA of the DI curve and compares it to the full
sample PCA. Saves both to a csv to be used in a backtesting routine.
"""

from pathlib import Path
from quantfin.data import DROPBOX
from sklearn.decomposition import PCA
from tqdm import tqdm
import getpass
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# User defined parameters
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 250)
start_date = '2023-03-01'  # TODO Correct the date


# Path to save outputs
username = getpass.getuser()
save_path = Path(f'/Users/{username}/Dropbox/Aulas/Insper - Renda Fixa/2023')

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
df_curve = df_curve.interpolate(axis=1, method='cubic')
df_curve = df_curve.dropna(how='any', axis=1)
df_curve.index = pd.to_datetime(df_curve.index)


# ===========================
# ===== Full sample PCA =====
# ===========================
pca_full = PCA(n_components=5)
pca_full.fit(df_curve.values)

df_var_full = pd.DataFrame(data=pca_full.explained_variance_ratio_)
df_loadings_full = pd.DataFrame(data=pca_full.components_.T,
                                columns=['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5'],
                                index=df_curve.columns)
df_mean_full = pd.DataFrame(data=pca_full.mean_, index=df_curve.columns,
                            columns=['Médias'])
df_pca_full = pd.DataFrame(data=pca_full.transform(df_curve.values),
                           columns=['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5'],
                           index=df_curve.index)

signal_full = np.sign(df_loadings_full.iloc[-1])
df_loadings_full = df_loadings_full * signal_full
df_pca_full = df_pca_full * signal_full


# ================================
# ===== Backtested PC Signal =====
# ================================
df_pca = pd.DataFrame(columns=['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5'])
df_loadings = pd.DataFrame(columns=['date', 'du', 'PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5'])
dates2loop = df_curve.index[df_curve.index >= start_date]

for date in tqdm(dates2loop, 'Generating Signal'):
    pca = PCA(n_components=5)
    data = df_curve.loc[:date]
    pca.fit(data.values)

    current_loadings = pd.DataFrame(data=pca.components_,
                                    index=['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5'],
                                    columns=df_curve.columns).T
    current_pca = pd.DataFrame(data=pca.transform(data.values),
                               columns=['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5'],
                               index=data.index)

    signal = np.sign(current_loadings.iloc[-1])
    current_loadings = current_loadings * signal
    current_pca = current_pca * signal

    df_pca.loc[date] = current_pca.iloc[-1]

    current_loadings = current_loadings.reset_index()
    current_loadings['date'] = date
    df_loadings = pd.concat([df_loadings, current_loadings])

# save data
df_loadings.to_csv(save_path.joinpath('DI1 PCA/expanding_loadings.csv'), sep=';')
df_pca.to_csv(save_path.joinpath('DI1 PCA/expanding_pcs.csv'), sep=';')

# ==================
# ===== Charts =====
# ==================
# ----- 1st PC -----
fig = plt.figure(figsize=(7 * (16 / 9), 7))
fig.suptitle('1st Principal Component', fontsize=16, fontweight='bold')

# Rolling VS Full Sample
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax1.plot(df_pca['PC 1'], label='Expanding Window')
ax1.plot(df_pca_full['PC 1'], label='Full Sample')
ax1.set_title('Estimation of the PCs')
ax1.grid(axis='y', alpha=0.3)
ax1.legend(frameon=True, loc='best')
ax1.axhline(0, color='black', linewidth=0.5)
locators = mdates.YearLocator()
ax1.xaxis.set_major_locator(locators)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.tick_params(rotation=90, axis='x')

# Evolution of Loadings
loadings_pc = df_loadings.pivot_table(values='PC 1', index='date', columns='du')
plot_load_start = loadings_pc.iloc[0]
plot_load_end = loadings_pc.iloc[-1]

ax2 = plt.subplot2grid((1, 2), (0, 1))
ax2.plot(plot_load_start.index, plot_load_start.values, label='Start of Sample')
ax2.plot(plot_load_end.index, plot_load_end.values, label='End of Sample')
ax2.set_title('Estimation of the Factor Loadings')
ax2.grid(axis='y', alpha=0.3)
ax2.set_xlabel('DU')
ax2.legend(frameon=True, loc='best')
ax2.axhline(0, color='black', linewidth=0.5)


plt.tight_layout()
plt.savefig(save_path.joinpath('figures/DI PC1 Expanding VS Full.pdf'))
plt.show()
plt.close()

# ----- 2nd PC -----
fig = plt.figure(figsize=(7 * (16 / 9), 7))
fig.suptitle('2nd Principal Component', fontsize=16, fontweight='bold')

# Rolling VS Full Sample
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax1.plot(df_pca['PC 2'], label='Expanding Window')
ax1.plot(df_pca_full['PC 2'], label='Full Sample')
ax1.set_title('Estimation of the PCs')
ax1.grid(axis='y', alpha=0.3)
ax1.legend(frameon=True, loc='best')
ax1.axhline(0, color='black', linewidth=0.5)
locators = mdates.YearLocator()
ax1.xaxis.set_major_locator(locators)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.tick_params(rotation=90, axis='x')

# Evolution of Loadings
loadings_pc = df_loadings.pivot_table(values='PC 2', index='date', columns='du')
plot_load_start = loadings_pc.iloc[0]
plot_load_end = loadings_pc.iloc[-1]

ax2 = plt.subplot2grid((1, 2), (0, 1))
ax2.plot(plot_load_start.index, plot_load_start.values, label='Start of Sample')
ax2.plot(plot_load_end.index, plot_load_end.values, label='End of Sample')
ax2.set_title('Estimation of the Factor Loadings')
ax2.grid(axis='y', alpha=0.3)
ax2.set_xlabel('DU')
ax2.legend(frameon=True, loc='best')
ax2.axhline(0, color='black', linewidth=0.5)


plt.tight_layout()
plt.savefig(save_path.joinpath('figures/DI PC2 Expanding VS Full.pdf'))
plt.show()
plt.close()


# ----- 3rd PC -----
fig = plt.figure(figsize=(7 * (16 / 9), 7))
fig.suptitle('3rd Principal Component', fontsize=16, fontweight='bold')

# Rolling VS Full Sample
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax1.plot(df_pca['PC 3'], label='Expanding Window')
ax1.plot(df_pca_full['PC 3'], label='Full Sample')
ax1.set_title('Estimation of the PCs')
ax1.grid(axis='y', alpha=0.3)
ax1.legend(frameon=True, loc='best')
ax1.axhline(0, color='black', linewidth=0.5)
locators = mdates.YearLocator()
ax1.xaxis.set_major_locator(locators)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.tick_params(rotation=90, axis='x')

# Evolution of Loadings
loadings_pc = df_loadings.pivot_table(values='PC 3', index='date', columns='du')
plot_load_start = loadings_pc.iloc[0]
plot_load_end = loadings_pc.iloc[-1]

ax2 = plt.subplot2grid((1, 2), (0, 1))
ax2.plot(plot_load_start.index, plot_load_start.values, label='Start of Sample')
ax2.plot(plot_load_end.index, plot_load_end.values, label='End of Sample')
ax2.set_title('Estimation of the Factor Loadings')
ax2.grid(axis='y', alpha=0.3)
ax2.set_xlabel('DU')
ax2.legend(frameon=True, loc='best')
ax2.axhline(0, color='black', linewidth=0.5)


plt.tight_layout()
plt.savefig(save_path.joinpath('figures/DI PC3 Expanding VS Full.pdf'))
plt.show()
plt.close()

# ----- 4th PC -----
fig = plt.figure(figsize=(7 * (16 / 9), 7))
fig.suptitle('4th Principal Component', fontsize=16, fontweight='bold')

# Rolling VS Full Sample
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax1.plot(df_pca['PC 4'], label='Expanding Window')
ax1.plot(df_pca_full['PC 4'], label='Full Sample')
ax1.set_title('Estimation of the PCs')
ax1.grid(axis='y', alpha=0.3)
ax1.legend(frameon=True, loc='best')
ax1.axhline(0, color='black', linewidth=0.5)
locators = mdates.YearLocator()
ax1.xaxis.set_major_locator(locators)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.tick_params(rotation=90, axis='x')

# Evolution of Loadings
loadings_pc = df_loadings.pivot_table(values='PC 4', index='date', columns='du')
plot_load_start = loadings_pc.iloc[0]
plot_load_end = loadings_pc.iloc[-1]

ax2 = plt.subplot2grid((1, 2), (0, 1))
ax2.plot(plot_load_start.index, plot_load_start.values, label='Start of Sample')
ax2.plot(plot_load_end.index, plot_load_end.values, label='End of Sample')
ax2.set_title('Estimation of the Factor Loadings')
ax2.grid(axis='y', alpha=0.3)
ax2.set_xlabel('DU')
ax2.legend(frameon=True, loc='best')
ax2.axhline(0, color='black', linewidth=0.5)


plt.tight_layout()
plt.savefig(save_path.joinpath('figures/DI PC4 Expanding VS Full.pdf'))
plt.show()
plt.close()


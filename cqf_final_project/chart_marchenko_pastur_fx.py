from quantfin.statistics import marchenko_pastur, targeted_shirinkage
from quantfin.portfolio import HRP
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

n_assets = 25
n_time = 5000
n_factors = 3
random_seed = 666

# file_path = r'/Users/gustavoamarante/Dropbox/CQF/Final Project'  # Mac
file_path = r'/Users/gusamarante/Dropbox/CQF/Final Project'  # Macbook

# Read Bloomberg Tickers for renaming
df_tickers = pd.read_excel(file_path + r'Data - BBG Data Values.xlsx',
                           index_col=0, sheet_name='Tickers')

tr_dict = df_tickers['Total Return Index (UBS)'].to_dict()
tr_dict = {v: k for k, v in tr_dict.items()}

# Read Total Return Index
df_tr = pd.read_excel(file_path + r'Data - BBG Data Values.xlsx',
                      index_col=0, sheet_name='Total Return')
data = df_tr.rename(tr_dict, axis=1)

# ===== plot correlation matrix =====
# hrp = HRP(data.pct_change(1))
# hrp.plot_corr_matrix(cmap='mako', figsize=(10, 10), show_chart=True,
#                      save_path=file_path + r'/figures/Simulated Correlation Matrix.pdf')
#
# # ===== compute different correlations matrices =====
# corr = data.corr().values  # empirical correlation
# corr_denoised, _, _ = marchenko_pastur(corr_matrix=corr, T=data.shape[0], N=data.shape[1])  # denoised correlation
# corr_ts, _, _ = targeted_shirinkage(corr_matrix=corr, T=data.shape[0], N=data.shape[1], ts_alpha=1)  # target-shrinkage
#
# # ===== sorted eigenvalues of each method =====
# eig_empirical = np.sort(np.linalg.eig(corr)[0])[::-1]
# eig_denoised = np.sort(np.linalg.eig(corr_denoised)[0])[::-1]
# eig_ts = np.sort(np.linalg.eig(corr_ts)[0])[::-1]
# df_eig = pd.DataFrame(data={'Empirical Eigenvalues': eig_empirical,
#                             'Denoised Eigenvalues': eig_denoised,
#                             'Targeted Shirinkage': eig_ts},
#                       index=[i + 1 for i in range(n_assets)])
#
# # ===== chart =====
# # time series Chart
# MyFont = {'fontname': 'Century Gothic'}
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Century Gothic']
#
# plt.figure(figsize=(10, 4))
# plt.plot(df_eig, linewidth=2)
# plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=True)
# plt.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=True)
# plt.legend(df_eig.columns, loc='upper right', ncol=1, frameon=True)
# ax = plt.gca()
# ax.set(yscale='log', xlabel='Sorted Eigenvalues', ylabel='Log-Scale')
# ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
# ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
# plt.tight_layout()
# plt.savefig(file_path + f'/figures/Simulated Sorted Eigenvalues.pdf', pad_inches=0)
# plt.show()
#
# # ===== get characteristics =====
# df_char = pd.DataFrame()
# corr_mat_dict = {'Empirical': corr,
#                  'Denoised': corr_denoised,
#                  'Targeted Shirinkage': corr_ts}
# for mat in corr_mat_dict.keys():
#     eigvals = np.linalg.eig(corr_mat_dict[mat])[0]
#     df_char.loc['Maximum Eigenvalue', mat] = np.max(eigvals)
#     df_char.loc['Minimum Eigenvalue', mat] = np.min(eigvals)
#     df_char.loc['Determinant', mat] = np.linalg.det(corr_mat_dict[mat])
#     df_char.loc['Max Norm', mat] = np.max(np.abs(corr_mat_dict[mat] - corr))
#     df_char.loc['Frobenius Norm', mat] = np.linalg.norm(corr_mat_dict[mat] - corr, 'fro')
#
# df_char.to_clipboard()

print(data.dropna())
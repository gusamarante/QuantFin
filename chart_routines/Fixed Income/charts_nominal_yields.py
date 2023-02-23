"""
Essa rotina faz o gráfico dos yields da última curva de juros disponível para os títulos nominais.
"""
from quantfin.data import DROPBOX
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import getpass

# ===== Parameters =====
last_year = 2023
size = 8
ratio = 9.8 / 13.7

username = getpass.getuser()
save_path = Path(f'/Users/{username}/Dropbox/Aulas/Insper - Renda Fixa/2023/figures')

MyFont = {'fontname': 'Century Gothic'}
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Century Gothic']

# ===== LTN =====
# grab data
ltn = pd.DataFrame()

for year in tqdm(range(2003, last_year + 1), 'Reading files'):
    aux = pd.read_csv(DROPBOX.joinpath(f'trackers/dados_ltn {year}.csv'), sep=';')
    ltn = pd.concat([ltn, aux])

ltn['reference date'] = pd.to_datetime(ltn['reference date'])
ltn['maturity'] = pd.to_datetime(ltn['maturity'])

# Filter Last Date
last_date = ltn['reference date'].max()
df_last = ltn[ltn['reference date'] == last_date]
df_last = df_last.sort_values(by='maturity')

# plot curve
plt.figure(figsize=(size, size * ratio))
plt.scatter(x=df_last['du'],
            y=df_last['yield'],
            edgecolors=None,
            s=60,
            c='#3333ac')

plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=True)
plt.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=True)

plt.xlabel('Maturity (DU)', MyFont)
plt.ylabel('Yield', MyFont)

plt.title(f"Yield de LTNs em {last_date.strftime('%d/%b/%y')}", **MyFont)

ax = plt.gca()
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig(save_path.joinpath('LTN example curve.pdf'))
plt.show()

# ===== NTN-F =====
# grab data
ntnf = pd.DataFrame()

for year in tqdm(range(2003, last_year + 1), 'Reading files'):
    aux = pd.read_csv(DROPBOX.joinpath(f'trackers/dados_ntnf {year}.csv'), sep=';')
    ntnf = pd.concat([ntnf, aux])

ntnf['reference date'] = pd.to_datetime(ntnf['reference date'])
ntnf['maturity'] = pd.to_datetime(ntnf['maturity'])

# Filter Last Date
last_date = ntnf['reference date'].max()
df_last = ntnf[ntnf['reference date'] == last_date]
df_last = df_last.sort_values(by='maturity')

# plot curve
plt.figure(figsize=(size, size * ratio))
plt.scatter(x=df_last['du'],
            y=df_last['yield'],
            edgecolors=None,
            s=60,
            c='#3333ac')

plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=True)
plt.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=True)

plt.xlabel('Maturity (DU)', MyFont)
plt.ylabel('Yield', MyFont)

plt.title(f"Yield de NTNFs em {last_date.strftime('%d/%b/%y')}", **MyFont)

ax = plt.gca()
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig(save_path.joinpath('NTN-F example curve.pdf'))
plt.show()

from quantfin.data import grab_connection
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import pandas as pd

# ===== Parameters =====
size = 9
ratio = 9.8 / 13.7
date_format = '%d/%b/%y'
save_path = Path('/Users/gustavoamarante/Dropbox/Aulas/Insper - Renda Fixa/2022/figures')

MyFont = {'fontname': 'Century Gothic'}
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Century Gothic']

# ===== Connection =====
conn = grab_connection()

# ===== LTN =====
# grab data
query = 'SELECT * FROM raw_tesouro_direto WHERE bond_type=="Tesouro Prefixado"'
df_ltn = pd.read_sql(sql=query, con=conn)

# Filter Last Date
latest_date = df_ltn['reference_date'].max()
df_ltn = df_ltn[df_ltn['reference_date'] == latest_date]
df_ltn = df_ltn.sort_values(by='maturity')

# plot
plt.figure(figsize=(size, size * ratio))
plt.scatter(x=pd.to_datetime(df_ltn['maturity']),
            y=df_ltn['taxa_compra'],
            edgecolors=None,
            s=60,
            c='#3333ac')

plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=True)
plt.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=True)

plt.xlabel('Maturity', MyFont)
plt.ylabel('Yield', MyFont)

plt.title(f"Curva LTN em {pd.to_datetime(latest_date).strftime('%d/%b/%y')}", **MyFont)

ax = plt.gca()

ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))

plt.tight_layout()
plt.savefig(save_path.joinpath('LTN example curve.pdf'))
plt.show()


a = 1
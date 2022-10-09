"""
This routine grabs the trackers built and adds them to my database to easily handle them later.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quantfin.data import DROPBOX
from quantfin.data import tracker_uploader

df = pd.DataFrame()
file_path = DROPBOX.joinpath('trackers')

# ======================
# ===== IDA ANBIMA =====
# ======================
aux = pd.read_excel(file_path.joinpath('IDA Anbima.xlsx'), index_col=0, sheet_name='Sheet1', skiprows=3)
aux = aux.loc[aux.index.drop([np.nan, 'Dates'])]
aux = aux.astype(float)
aux.index = pd.to_datetime(aux.index)

aux = aux.rename({'IDADGRAL Index': 'IDA Geral',
                  'IDADDI Index': 'IDA DI',
                  'IDADIPCA Index': 'IDA IPCA',}, axis=1)

df = pd.concat([df, aux], axis=1)

# ============================
# ===== Tesouro Nacional =====
# ============================
# LFT Curta
aux = pd.read_csv(file_path.joinpath('lft_curta.csv'),
                  index_col=0, sep=';')
aux = aux['Notional'].rename('LFT Curta')
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# LFT Longa
aux = pd.read_csv(file_path.joinpath('lft_longa.csv'),
                  index_col=0, sep=';')
aux = aux['Notional'].rename('LFT Longa')
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# LTN Curta
aux = pd.read_csv(file_path.joinpath('ltn_curta.csv'),
                  index_col=0, sep=';')
aux = aux['Notional'].rename('LTN Curta')
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# LTN Longa
aux = pd.read_csv(file_path.joinpath('ltn_longa.csv'),
                  index_col=0, sep=';')
aux = aux['Notional'].rename('LTN Longa')
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# NTN-B Curta
aux = pd.read_csv(file_path.joinpath('ntnb_curta.csv'),
                  index_col=0, sep=';')
aux = aux['Notional'].rename('NTNB Curta')
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# NTN-B 2y
aux = pd.read_csv(file_path.joinpath('ntnb_2y.csv'),
                  index_col=0, sep=';')
aux = aux['Notional'].rename('NTNB 2y')
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# NTN-B 5y
aux = pd.read_csv(file_path.joinpath('ntnb_5y.csv'),
                  index_col=0, sep=';')
aux = aux['Notional'].rename('NTNB 5y')
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# NTN-B 10y
aux = pd.read_csv(file_path.joinpath('ntnb_10y.csv'),
                  index_col=0, sep=';')
aux = aux['Notional'].rename('NTNB 10y')
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# NTN-B Longa
aux = pd.read_csv(file_path.joinpath('ntnb_longa.csv'),
                  index_col=0, sep=';')
aux = aux['Notional'].rename('NTNB Longa')
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# NTN-F Curta
aux = pd.read_csv(file_path.joinpath('ntnf_curta.csv'),
                  index_col=0, sep=';')
aux = aux['Notional'].rename('NTNF Curta')
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# NTN-F 2y
aux = pd.read_csv(file_path.joinpath('ntnf_2y.csv'),
                  index_col=0, sep=';')
aux = aux['Notional'].rename('NTNF 2y')
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# NTN-F 5y
aux = pd.read_csv(file_path.joinpath('ntnf_5y.csv'),
                  index_col=0, sep=';')
aux = aux['Notional'].rename('NTNF 5y')
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# NTN-B Longa
aux = pd.read_csv(file_path.joinpath('ntnf_longa.csv'),
                  index_col=0, sep=';')
aux = aux['Notional'].rename('NTNF Longa')
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)


# ===========================
# ===== Fundo Exclusivo =====
# ===========================
# BDIV
aux = pd.read_excel(file_path.joinpath('BDIV.xlsx'),
                    index_col=0, sheet_name='Tracker')
aux = aux['BDIV']
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# JURO
aux = pd.read_excel(file_path.joinpath('JURO.xlsx'),
                    index_col=0, sheet_name='Tracker')
aux = aux['JURO']
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)


# ======================
# ===== ETFs da B3 =====
# ======================
aux = pd.read_excel(file_path.joinpath('ETFs B3.xlsx'),
                    index_col=0, sheet_name='Trackers')
aux.columns = aux.columns.str[:4]
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)


# ======================
# ===== Cota da XP =====
# ======================
aux = pd.read_excel(file_path.joinpath('XP.xlsx'),
                    index_col=0, sheet_name='day')
aux = aux['tracker'].rename('Cota XP')
df = pd.concat([df, aux], axis=1)


# =============================
# ===== Save the Trackers =====
# =============================
tracker_uploader(df)


# =============================
# ===== Plot the Trackers =====
# =============================
print(df.index[-1])
df.plot()
plt.show()

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

# NTN-Bs
available_ntnb = ['0p5', '1', '1p5', '2', '3', '4', '5', '6', '7', '8', '9', '10', '15', '20', '25']  # TODO how to deal with the duplicate 15

for mat in available_ntnb:

    aux = pd.read_csv(file_path.joinpath(f'ntnb_{mat}y.csv'),
                      index_col=0, sep=';')

    tracker_name = f'NTNB {mat.replace("p", ".")}y'
    aux = aux['Notional'].rename(tracker_name)
    aux.index = pd.to_datetime(aux.index)
    df = pd.concat([df, aux], axis=1)

# NTN-Fs
available_ntnf = ['05', '1', '15', '2', '3', '4', '5']

for mat in available_ntnf:

    aux = pd.read_csv(file_path.joinpath(f'ntnf_{mat}y.csv'),
                      index_col=0, sep=';')

    if mat in ['05', '15']:
        tracker_name = f'NTNF {mat[0]}.{mat[1]}y'
    else:
        tracker_name = f'NTNF {mat}y'

    aux = aux['Notional'].rename(tracker_name)
    aux.index = pd.to_datetime(aux.index)
    df = pd.concat([df, aux], axis=1)


# ===================================================
# ===== Fundos de Investimento em Participações =====
# ===================================================
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

# XPIE
aux = pd.read_excel(file_path.joinpath('XPIE.xlsx'),
                    index_col=0, sheet_name='Tracker')
aux = aux['XPIE']
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

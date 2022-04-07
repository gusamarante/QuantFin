"""
This routine grabs the trackers built and adds them to my database to easily handle them later.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from quantfin.data import tracker_uploader

df = pd.DataFrame()
file_path = Path(r"C:\Users\gamarante\Dropbox\Personal Portfolio\trackers")  # BW
# file_path = Path(r"/Users/gustavoamarante/Dropbox/Personal Portfolio/trackers")  # Mac

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

# NTN-B Longa
aux = pd.read_csv(file_path.joinpath('ntnf_longa.csv'),
                  index_col=0, sep=';')
aux = aux['Notional'].rename('NTNF Longa')
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)


# ======================
# ===== ETFs da B3 =====
# ======================
# BDIV11
aux = pd.read_excel(file_path.joinpath('BDIV11.xlsx'),
                    index_col=0, sheet_name='Tracker')
aux = aux['BDIV11']
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# BOVA11
aux = pd.read_excel(file_path.joinpath('BOVA11.xlsx'),
                    index_col=0, sheet_name='Tracker')
aux = aux['BOVA11']
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# HASH11
aux = pd.read_excel(file_path.joinpath('HASH11.xlsx'),
                    index_col=0, sheet_name='Tracker')
aux = aux['HASH11']
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# HASH11
aux = pd.read_excel(file_path.joinpath('SPXI11.xlsx'),
                    index_col=0, sheet_name='Tracker')
aux = aux['SPXI11']
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
df.plot()
plt.show()

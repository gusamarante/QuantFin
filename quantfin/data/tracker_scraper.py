"""
This routine grabs the trackers built and adds them to my database to easily handle them later.
"""

import pandas as pd
import matplotlib.pyplot as plt
from quantfin.data import tracker_uploader

df = pd.DataFrame()

# ============================
# ===== Tesouro Nacional =====
# ============================
# LFT Curta
aux = pd.read_csv('/Users/gustavoamarante/Dropbox/Personal Portfolio/trackers/lft_curta.csv',
                  index_col=0, sep=';')
aux = aux['Notional'].rename('LFT Curta')
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# LFT Longa
aux = pd.read_csv('/Users/gustavoamarante/Dropbox/Personal Portfolio/trackers/lft_longa.csv',
                  index_col=0, sep=';')
aux = aux['Notional'].rename('LFT Longa')
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# NTN-B Curta
aux = pd.read_csv('/Users/gustavoamarante/Dropbox/Personal Portfolio/trackers/ntnb_curta.csv',
                  index_col=0, sep=';')
aux = aux['Notional'].rename('NTNB Curta')
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# NTN-B Longa
aux = pd.read_csv('/Users/gustavoamarante/Dropbox/Personal Portfolio/trackers/ntnb_longa.csv',
                  index_col=0, sep=';')
aux = aux['Notional'].rename('NTNB Longa')
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)


# ======================
# ===== ETFs da B3 =====
# ======================
# BDIV11
aux = pd.read_excel('/Users/gustavoamarante/Dropbox/Personal Portfolio/trackers/BDIV11.xlsx',
                    index_col=0, sheet_name='Tracker')
aux = aux['BDIV11']
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# BOVA11
aux = pd.read_excel('/Users/gustavoamarante/Dropbox/Personal Portfolio/trackers/BOVA11.xlsx',
                    index_col=0, sheet_name='Tracker')
aux = aux['BOVA11']
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# HASH11
aux = pd.read_excel('/Users/gustavoamarante/Dropbox/Personal Portfolio/trackers/HASH11.xlsx',
                    index_col=0, sheet_name='Tracker')
aux = aux['HASH11']
aux.index = pd.to_datetime(aux.index)
df = pd.concat([df, aux], axis=1)

# HASH11
aux = pd.read_excel('/Users/gustavoamarante/Dropbox/Personal Portfolio/trackers/SPXI11.xlsx',
                    index_col=0, sheet_name='Tracker')
aux = aux['SPXI11']
aux.index = pd.to_datetime(aux.index)
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

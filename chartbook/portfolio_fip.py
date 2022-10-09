"""
This routine builds the optimal FIP (Fundo de investimento em participações) portfolio.
The available assets for this portfolio are:
    - BDIV11: Infra / BTG
    - JURO11: Infra / Sparta
"""
from quantfin.data import tracker_feeder, SGS, DROPBOX
from quantfin.finmath import compute_eri

# Benchmark
sgs = SGS()
df_cdi = sgs.fetch({12: 'CDI'})
df_cdi = df_cdi['CDI'] / 100

# Trackers
df_tri = tracker_feeder()
df_tri = df_tri[['BDIV', 'JURO']]

# Excess Returns
df_eri = compute_eri(df_tri, df_cdi)

# Construction Method
# TODO EW
# TODO ERC
# TODO HRP

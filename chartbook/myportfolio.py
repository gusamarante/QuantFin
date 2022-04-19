from quantfin.charts import timeseries, df2pdf, df2heatmap
from quantfin.data import tracker_feeder, SGS
from quantfin.portfolio import Performance
from quantfin.finmath import compute_eri
import matplotlib.pyplot as plt
from pathlib2 import Path
import pandas as pd

# Grab data
df_tri = tracker_feeder()
df_tri = df_tri.drop(['Cota XP'], axis=1)
df_tri = df_tri[df_tri.index >= '2008-01-01']

# Risk-free
sgs = SGS()
df_cdi = sgs.fetch({12: 'CDI'})
df_cdi = df_cdi['CDI'] / 100

# Compute ERI
df_eri = compute_eri(df_tri, df_cdi)

# Performance Data
perf_t = Performance(df_eri)

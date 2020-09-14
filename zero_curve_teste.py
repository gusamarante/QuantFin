from quantfin.assets import ZeroCurve
import pandas as pd


# Mac at home
# df = pd.read_excel(r'/Users/gustavoamarante/Dropbox/Aulas/QuantFin/Tracker Building/CDS Sample Data.xlsx',
#                    sheet_name='CurveValues', index_col='Dates', na_values=['#N/A N/A'])

# Macbook
df = pd.read_excel(r'/Users/gusamarante/Dropbox/Aulas/QuantFin/Tracker Building/CDS Sample Data.xlsx',
                   sheet_name='CurveValues', index_col='Dates', na_values=['#N/A N/A'])

rename_dict = {'USLFD1M  ISCF Curncy': 1/12,
               'USLFD2M  ISCF Curncy': 2/12,
               'USLFD3M  ISCF Curncy': 3/12,
               'USLFD6M  ISCF Curncy': 6/12,
               'USLFD12M ISCF Curncy': 12/12,
               'USSWAP2  ISCF Curncy': 24/12,
               'USSWAP3  ISCF Curncy': 36/12,
               'USSWAP4  ISCF Curncy': 48/12,
               'USSWAP5  ISCF Curncy': 60/12,
               'USSWAP6  ISCF Curncy': 72/12,
               'USSWAP7  ISCF Curncy': 84/12,
               'USSWAP8  ISCF Curncy': 96/12,
               'USSWAP9  ISCF Curncy': 108/12,
               'USSWAP10 ISCF Curncy': 120/12,
               'USSWAP12 ISCF Curncy': 144/12,
               'USSWAP15 ISCF Curncy': 180/12,
               'USSWAP20 ISCF Curncy': 240/12,
               'USSWAP25 ISCF Curncy': 300/12,
               'USSWAP30 ISCF Curncy': 360/12}

df = df.rename(rename_dict, axis=1)
df = df.dropna(how='all')
df = df.div(100)

zc = ZeroCurve(df, 'act/360')

xxx = zc.rate(1.2, '2020-09-10')

print(xxx)

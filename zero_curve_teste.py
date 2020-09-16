from quantfin.assets import ZeroCurve
import pandas as pd


# Mac at home
# df = pd.read_excel(r'/Users/gustavoamarante/Dropbox/Aulas/QuantFin/Tracker Building/CDS Sample Data.xlsx',
#                    sheet_name='CurveValues', index_col='Dates', na_values=['#N/A N/A'])

# Macbook
df = pd.read_excel(r'/Users/gusamarante/Dropbox/Aulas/QuantFin/Tracker Building/CDS Sample Data.xlsx',
                   sheet_name='CurveValues', index_col='Dates', na_values=['#N/A N/A'])

rename_dict = {'USLFD1M  ISCF Curncy': 1*30,
               'USLFD2M  ISCF Curncy': 2*30,
               'USLFD3M  ISCF Curncy': 3*30,
               'USLFD6M  ISCF Curncy': 6*30,
               'USLFD12M ISCF Curncy': 12*30,
               'USSWAP2  ISCF Curncy': 24*30,
               'USSWAP3  ISCF Curncy': 36*30,
               'USSWAP4  ISCF Curncy': 48*30,
               'USSWAP5  ISCF Curncy': 60*30,
               'USSWAP6  ISCF Curncy': 72*30,
               'USSWAP7  ISCF Curncy': 84*30,
               'USSWAP8  ISCF Curncy': 96*30,
               'USSWAP9  ISCF Curncy': 108*30,
               'USSWAP10 ISCF Curncy': 120*30,
               'USSWAP12 ISCF Curncy': 144*30,
               'USSWAP15 ISCF Curncy': 180*30,
               'USSWAP20 ISCF Curncy': 240*30,
               'USSWAP25 ISCF Curncy': 300*30,
               'USSWAP30 ISCF Curncy': 360*30}

df = df.rename(rename_dict, axis=1)
df = df.dropna(how='all')
df = df.div(100)

zc = ZeroCurve(df, 'act/360')

xxx = zc.forward(360*1.2, 360*1.5, '2020-09-10')

print(xxx)

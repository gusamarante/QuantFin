from quantfin.finmath import ZeroCurve
import pandas as pd


# Mac at home
# df = pd.read_excel(r'/Users/gustavoamarante/Dropbox/Aulas/QuantFin/Tracker Building/CDS Sample Data.xlsx',
#                    sheet_name='CurveValues', index_col='Dates', na_values=['#N/A N/A'])

# Macbook
df = pd.read_excel(r'/Users/gusamarante/Dropbox/Aulas/QuantFin/Tracker Building/CDS Sample Data.xlsx',
                   sheet_name='CurveValues', index_col='Dates', na_values=['#N/A N/A'])
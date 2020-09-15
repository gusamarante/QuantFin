import pandas as pd

# Mac at home
df = pd.read_excel(r'/Users/gustavoamarante/Dropbox/Aulas/QuantFin/Tracker Building/CDS Sample Data.xlsx',
                   sheet_name='CDSValues', index_col='Dates', na_values=['#N/A N/A'])

# Macbook
# df = pd.read_excel(r'/Users/gusamarante/Dropbox/Aulas/QuantFin/Tracker Building/CDS Sample Data.xlsx',
#                    sheet_name='CDSValues', index_col='Dates', na_values=['#N/A N/A'])

rename_dict = {'BRAZIL CDS USD SR 0M D14 Corp': 30*0,
               'BRAZIL CDS USD SR 3M D14 Corp': 30*3,
               'BRAZIL CDS USD SR 6M D14 Corp': 30*6,
               'BRAZIL CDS USD SR 9M D14 Corp': 30*9,
               'BRAZIL CDS USD SR 1Y D14 Corp': 30*12,
               'BRAZIL CDS USD SR 2Y D14 Corp': 30*24,
               'BRAZIL CDS USD SR 3Y D14 Corp': 30*36,
               'BRAZIL CDS USD SR 4Y D14 Corp': 30*48,
               'BRAZIL CDS USD SR 5Y D14 Corp': 30*60,
               'BRAZIL CDS USD SR 6Y D14 Corp': 30*72,
               'BRAZIL CDS USD SR 7Y D14 Corp': 30*84,
               'BRAZIL CDS USD SR 8Y D14 Corp': 30*96,
               'BRAZIL CDS USD SR 9Y D14 Corp': 30*108,
               'BRAZIL CDS USD SR 10Y D14 Corp': 30*120,
               'BRAZIL CDS USD SR 11Y D14 Corp': 30*132,
               'BRAZIL CDS USD SR 12Y D14 Corp': 30*144,
               'BRAZIL CDS USD SR 15Y D14 Corp': 30*180,
               'BRAZIL CDS USD SR 20Y D14 Corp': 30*240,
               'BRAZIL CDS USD SR 30Y D14 Corp': 30*360}

df = df.rename(rename_dict, axis=1)
df = df.dropna(how='all', axis=1)



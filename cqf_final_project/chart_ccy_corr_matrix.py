import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ccy_g10 = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK']
ccy_em = ['BRL', 'CLP', 'CZK', 'HUF', 'IDR', 'ILS', 'INR', 'KRW', 'MXN', 'PHP',
          'PLN', 'RUB', 'SGD', 'TRY', 'TWD', 'ZAR']
ccy_all = ccy_em + ccy_g10

df = pd.read_excel(r'/Users/gustavoamarante/Dropbox/CQF/Final Project/Data - Spot Rates.xlsx',
                   index_col=0)

df = df[ccy_all]

df = df.pct_change(1)
df = df[df.index >= '2017-01-01']
corr = df.pct_change(1).corr()

# Chart
print(corr)
sns.clustermap(data=corr, method='average', metric='euclidean', row_cluster=False, col_cluster=True)
plt.show()


a = 1

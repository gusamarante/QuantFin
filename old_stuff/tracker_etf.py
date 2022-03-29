import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

bdvi = yf.Ticker('SPXI11.SA')
df_hist = bdvi.history(period='max')

print(df_hist)

df_hist['Close'].plot()
plt.show()

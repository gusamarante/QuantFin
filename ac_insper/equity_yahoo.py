"""
Yahoo penalizes if you make too many queries too fast.
"""
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt

# TODO turn this into a function

# Read data from yahoo finance
df_tracker = pdr.DataReader('RENT3.SA', 'yahoo')
dividend = pdr.DataReader('RENT3.SA', 'yahoo-dividends')

# Build the tracker
df_tracker['Div'] = dividend['value']
df_tracker['Div'] = df_tracker['Div'].fillna(0)

df_tracker['New Shares'] = df_tracker['Div'] / df_tracker['High']  # Estimate of trading costs (bad execution)
df_tracker['Total Shares'] = df_tracker['New Shares'].cumsum() + 1
df_tracker['Notional'] = df_tracker['Close'] * df_tracker['Total Shares']

# Chart
df_tracker[['Close', 'Notional']].plot()
plt.show()


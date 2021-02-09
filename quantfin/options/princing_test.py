import numpy as np
import matplotlib.pyplot as plt
from quantfin.options import BinomalTree

bt = BinomalTree(stock=100,
                 strike=90,
                 years2mat=1,
                 vol=0.20,
                 risk_free=0.05,
                 div_yield=0,
                 n=4,
                 call=False,
                 option_type='american')

# Price
print(bt.price)

# Chart the tree for the stock price
fig = bt.chart_stock(labels=True)
plt.show()


# ===== Convergence of the price =====
def option_price(n=1):
    return BinomalTree(stock=100, strike=90, years2mat=1, vol=0.20, risk_free=0.05, n=n).price


n_grid = np.arange(1, 101, 1, dtype=int)

plt.figure()
plt.plot(n_grid, list(map(option_price, n_grid)))
plt.xlabel('Number of steps in the binomial tree')
plt.ylabel('option price')
plt.show()

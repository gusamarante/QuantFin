from quantfin.statistics import GaussianHMM
from quantfin.portfolio import DAACosts
from hmmlearn import hmm
import pandas as pd
import numpy as np

notional = 100

# ===== Simulate Data =====
transmat = pd.DataFrame(data=np.array([[0.5, 0.5],
                                       [0.5, 0.5]]),
                        index=['From State 1', 'From State 2'],
                        columns=['To State 1', 'To State 2'])

means = pd.DataFrame(data=np.array([[0.05, 0.03],
                                    [0.03, 0.05]]),
                     index=['State 1', 'State 2'],
                     columns=['Asset 1', 'Asset 2'])

mindex = pd.MultiIndex.from_product([['State 1', 'State 2'], ['Asset 1', 'Asset 2']])
covars = pd.DataFrame(index=mindex, columns=['Asset 1', 'Asset 2'],
                      data=np.array([[0.01, 0.0],
                                     [0.0, 0.01],
                                     [0.01, 0.0],
                                     [0.0, 0.01]]))

# ===== Compute Allocations =====
start_alloc = notional * np.array([1.50, 1.50])
Lambda1 = (1 / 10000) * np.eye(2)
Lambda2 = (1 / 10000) * np.eye(2)
Lambda = np.array([Lambda1, Lambda2])

daa = DAACosts(means=means,
               covars=covars,
               costs=Lambda,
               transition_matrix=transmat,
               current_allocation=start_alloc,
               risk_aversion=0.0266667431442302,
               discount_factor=0.99,
               include_returns=True,
               normalize=False)

print('Model Allocations')
print(daa.allocations, '\n')

print('Aim Portfolios')
print(daa.aim_portfolios, '\n')

print('Mkw Portfolios')
print(daa.markowitz_portfolios, '\n')


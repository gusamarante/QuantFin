"""
2 ativos e 4 estados
"""

from quantfin.portfolio import DAACosts
import pandas as pd
import numpy as np

notional = 100

# ===== Simulate Data =====
transmat = pd.DataFrame(data=np.array([[0.86, 0.04, 0, 0.1],
                                       [0.013, 0.544, 0.148, 0.295],
                                       [0, 0.13, 0.433, 0.437],
                                       [0.673, 0.167, 0, 0.16]]),
                        index=['From State 1', 'From State 2', 'From State 3', 'From State 4'],
                        columns=['To State 1', 'To State 2', 'To State 3', 'To State 4'])

means = pd.DataFrame(data=np.array([[0.002522, 0.007863],
                                    [0.036255, -0.033534],
                                    [0.016122, -0.021536],
                                    [-0.012106, 0.016015]]),
                     index=['State 1', 'State 2', 'State 3', 'State 4'],
                     columns=['Asset 1', 'Asset 2'])

mindex = pd.MultiIndex.from_product([['State 1', 'State 2', 'State 3', 'State 4'], ['Asset 1', 'Asset 2']])
covars = pd.DataFrame(index=mindex, columns=['Asset 1', 'Asset 2'],
                      data=np.array([
                          [0.000446, -0.000006],
                          [-0.000006, 0.000368],

                          [0.001649, 0.000451],
                          [0.000145, 0.000977],

                          [0.00184, 0.002305],
                          [0.002305, 0.003295],

                          [0.001790, 0.000103],
                          [0.000103, 0.001006]
                      ]))

# ===== Compute Allocations =====
start_alloc = 10000000000 * np.array([0.25, 0.25])

Lambda1 = (1 / 10000) * np.eye(2)
Lambda2 = (1 / 10000) * np.eye(2)
Lambda3 = (1 / 10000) * np.eye(2)
Lambda4 = (1 / 10000) * np.eye(2)
Lambda = np.array([Lambda1, Lambda2, Lambda3, Lambda4])

daa = DAACosts(means=means,
               covars=covars,
               costs=Lambda,
               transition_matrix=transmat,
               current_allocation=start_alloc,
               risk_aversion=1.08679386478568e-9,
               discount_factor=0.99,
               include_returns=True,
               normalize=True)

print('Model Allocations')
print(daa.allocations, '\n')

print('Aim Portfolios')
print(daa.aim_portfolios, '\n')

print('Mkw Portfolios')
print(daa.markowitz_portfolios, '\n')

from quantfin.portfolio import DAACosts
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

notional = 100000000

for p in [0, 0.3, 0.6]:
    # ===== Simulate Data =====
    transmat = pd.DataFrame(data=np.array([[0.80, 0.15, 0.05, 0.0],
                                           [0.10, 0.60, 0.20, 0.1],
                                           [0.05, 0.15, 0.50, 0.3],
                                           [0.7 - p, 0.15, 0.15, p]]),
                            index=['From State 1', 'From State 2', 'From State 3', 'From State 4'],
                            columns=['To State 1', 'To State 2', 'To State 3', 'To State 4'])

    mindex = pd.MultiIndex.from_product([['State 1', 'State 2', 'State 3', 'State 4'], ['Asset 1', 'Asset 2']])
    covars = pd.DataFrame(index=mindex, columns=['Asset 1', 'Asset 2'],
                          data=np.array([
                              [0.01 / 12., 0.0],
                              [0.0, 0.01 / 12.],

                              [0.01 / 12., 0.0],
                              [0.0, 0.01 / 12.],

                              [0.01 / 12., 0.0],
                              [0.0, 0.01 / 12.],

                              [0.01 / 12., 0.0],
                              [0.0, 0.01 / 12.],
                          ]))

    sharpes = pd.DataFrame(data=np.array([[0.3, 0.3],
                                          [0.1, -0.1],
                                          [-0.1, 0.1],
                                          [-0.3, -0.3]]),
                           index=['State 1', 'State 2', 'State 3', 'State 4'],
                           columns=['Asset 1', 'Asset 2'])

    vols = pd.DataFrame(data=np.array([[np.sqrt(covars.iloc[0, 0]), np.sqrt(covars.iloc[1, 1])],
                                       [np.sqrt(covars.iloc[2, 0]), np.sqrt(covars.iloc[3, 1])],
                                       [np.sqrt(covars.iloc[4, 0]), np.sqrt(covars.iloc[5, 1])],
                                       [np.sqrt(covars.iloc[6, 0]), np.sqrt(covars.iloc[7, 1])]]),
                        index=['State 1', 'State 2', 'State 3', 'State 4'],
                        columns=['Asset 1', 'Asset 2'])

    means = sharpes * vols

    # ===== Compute Allocations =====
    start_alloc = notional * np.array([0.15, 0.15])

    Lambda1 = (1 / 100000000000) * np.eye(2)
    Lambda2 = (1 / 100000000000) * np.eye(2)
    Lambda3 = (3 / 100000000000) * np.eye(2)
    Lambda4 = (1 / 100000000000) * np.eye(2)
    Lambda = np.array([Lambda1, Lambda2, Lambda3, Lambda4])

    daa = DAACosts(means=means,
                   covars=covars,
                   costs=Lambda,
                   transition_matrix=transmat,
                   current_allocation=start_alloc,
                   risk_aversion=1.08679386478568e-9,
                   discount_factor=0.99,
                   include_returns=True,
                   normalize=False)

    plt.figure(figsize=(15, 10))
    plt.scatter(daa.allocations.iloc[:, 0].values, daa.allocations.iloc[:, 1].values,
                cmap=plt.get_cmap('RdBu'), alpha=0.4)

    plt.scatter(daa.aim_portfolios.iloc[:,0].values, daa.aim_portfolios.iloc[:,1].values,
                cmap=plt.get_cmap('RdBu'), alpha=0.6)

    plt.scatter(daa.markowitz_portfolios.iloc[:,0].values, daa.markowitz_portfolios.iloc[:,1].values,
                cmap=plt.get_cmap('RdBu'), alpha=0.9)

    plt.scatter(daa.unconditional_portfolio.iloc[0], daa.unconditional_portfolio.iloc[1])

    plt.title(f'Allocation across four states p={p * 100}%', fontsize=18)
    plt.ylabel('Asset 2 (Equities) allocation', fontsize=16)
    plt.xlabel('Asset 1 (Rates) allocation', fontsize=16)
    plt.legend(['Traded', 'Target', 'Markowitz', 'Static'])
    plt.show()



from quantfin.portfolio import Markowitz
import pandas as pd

# ===== 1 Risky Asset =====
mu = pd.Series(data={'A': 0.1},
               name='mu')

sigma = pd.Series(data={'A': 0.2},
                  name='sigma')

corr = [[1]]

corr = pd.DataFrame(columns=['A'],
                    index=['A'],
                    data=corr)

mkw = Markowitz(mu, sigma, corr,
                rf=0.02,
                risk_aversion=2.5)

# ===== 2 Risky Assets =====
mu = pd.Series(data={'A': 0.10,
                     'B': 0.15},
               name='mu')

sigma = pd.Series(data={'A': 0.2,
                        'B': 0.30},
                  name='sigma')

corr = [[1.0, 0.3],
        [0.3, 1.0]]

corr = pd.DataFrame(columns=['A', 'B'],
                    index=['A', 'B'],
                    data=corr)

mkw = Markowitz(mu, sigma, corr,
                rf=0.02,
                risk_aversion=2.5)

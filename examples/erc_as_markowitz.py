"""
The ideia of this routine is to show that there is a mu(sigma)
that turn the markowitz solution into the ERC solution.
"""
from quantfin.portfolio import MaxSharpe, ERC
from quantfin.statistics import corr2cov
import pandas as pd
import numpy as np


# Notice that we are not setting expected returns for the assets.
gamma = 1

corr = np.array([[1.0, 0.8, 0.0, 0.0],
                 [0.8, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, -0.5],
                 [0.0, 0.0, -0.5, 1.0]])
corr = pd.DataFrame(data=corr)

vols = np.array([0.1, 0.2, 0.3, 0.4])
vols = pd.Series(data=vols)

cov = corr2cov(corr, vols)

# Equal Weighted
weights_equal = 0.25 * np.ones(4)
vol_ew = np.sqrt(weights_equal.T @ cov.values @ weights_equal)
marginal_risk_ew = (weights_equal @ cov) / vol_ew
risk_contribution = marginal_risk_ew * weights_equal
risk_contribution_ratio = risk_contribution / vol_ew

# ERC
erc = ERC(cov)

print('weights \n', erc.weights)
print('vol', erc.vol)
print('Marginal risk \n', erc.marginal_risk)
print('Risk Contribution \n', erc.risk_contribution)
print('Risk Contribution Ratio \n', erc.risk_contribution_ratio)

mu_tilde = erc.marginal_risk * erc.vol * gamma

# Markowitz
ms = MaxSharpe(mu=mu_tilde, cov=cov, rf=0)
print('Max Sharpe \n', ms.risky_weights)

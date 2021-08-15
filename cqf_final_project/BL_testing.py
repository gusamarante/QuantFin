from quantfin.portfolio import BlackLitterman
import numpy as np
import pandas as pd

asset_list = ['Asset A', 'Asset B', 'Asset C', 'Asset D']
view_list = ['View 1', 'View 2']

corr = np.array([[1, 0.3, 0.1, 0.2],
                 [0.3, 1, 0.5, 0.2],
                 [0.1, 0.5, 1, -0.1],
                 [0.2, 0.2, -0.1, 1]])

vol = np.array([0.1, 0.15, 0.2, 0.18])

sigma = np.diag(vol) @ corr @ np.diag(vol)

sigma = pd.DataFrame(data=sigma, columns=asset_list, index=asset_list)

tau = 1/500

views_p = np.array([[1, 0, 0, 0],
                    [0, 0, -1, 1]])
views_p = pd.DataFrame(data=views_p, columns=asset_list, index=view_list)

views_v = np.array([0.12, 0.05])
views_v = pd.DataFrame(data=views_v, index=view_list, columns=['View Values'])

u = np.array([2, 1])
u = pd.DataFrame(data=u, index=view_list, columns=['Relative Uncertainty'])

w_equilibrium = np.array([0.25, 0.25, 0.25, 0.25])
w_equilibrium = pd.DataFrame(data=w_equilibrium, index=asset_list, columns=['Equilibrium Weights'])

mu_historical = np.array([0, 0, 0, 0])
mu_historical = pd.DataFrame(data=mu_historical, index=asset_list, columns=['Historical Returns'])

bl = BlackLitterman(sigma=sigma,
                    estimation_error=tau,
                    w_equilibrium=w_equilibrium,
                    avg_risk_aversion=1.2,
                    mu_shrink=0.9,
                    views_p=views_p,
                    views_v=views_v,
                    overall_confidence=1.5,
                    relative_uncertainty=u,
                    mu_historical=mu_historical)

print(bl.mu_bl)
print(bl.sigma_bl)

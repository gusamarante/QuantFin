from quantfin.portfolio import BlackLitterman, Markowitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_path = r'/Users/gustavoamarante/Dropbox/CQF/Final Project'  # Mac
# file_path = r'/Users/gusamarante/Dropbox/CQF/Final Project'  # Macbook

asset_list = ['Asset A', 'Asset B', 'Asset C', 'Asset D']
view_list = ['View 1', 'View 2']

# Covariance of returns
corr = np.array([[1, 0.3, 0.1, 0.2],
                 [0.3, 1, 0.5, 0.2],
                 [0.1, 0.5, 1, -0.1],
                 [0.2, 0.2, -0.1, 1]])

vol = np.array([0.1, 0.12, 0.15, 0.18])
sigma = np.diag(vol) @ corr @ np.diag(vol)

corr = pd.DataFrame(data=corr, columns=asset_list, index=asset_list)
vol = pd.Series(data=vol, index=asset_list, name='Vol')
sigma = pd.DataFrame(data=sigma, columns=asset_list, index=asset_list)

# Views
tau = 1/500

views_p = np.array([[1, 0, 0, 0],
                    [0, 0, 0, 1]])
views_p = pd.DataFrame(data=views_p, columns=asset_list, index=view_list)

views_v = np.array([0.2, 0.15])
views_v = pd.DataFrame(data=views_v, index=view_list, columns=['View Values'])

u = np.array([1, 1])
u = pd.DataFrame(data=u, index=view_list, columns=['Relative Uncertainty'])

# best guess for mu
w_equilibrium = np.array([0.25, 0.25, 0.25, 0.25])
w_equilibrium = pd.DataFrame(data=w_equilibrium, index=asset_list, columns=['Equilibrium Weights'])

mu_historical = np.array([0, 0, 0, 0])
mu_historical = pd.DataFrame(data=mu_historical, index=asset_list, columns=['Historical Returns'])

bl = BlackLitterman(sigma=sigma,
                    estimation_error=tau,
                    w_equilibrium=w_equilibrium,
                    avg_risk_aversion=1.2,
                    mu_shrink=1,
                    views_p=views_p,
                    views_v=views_v,
                    overall_confidence=10,
                    relative_uncertainty=u,
                    mu_historical=mu_historical)

mkw_original = Markowitz(bl.mu_best_guess['Best Guess of mu'], vol, corr, 0.01, risk_aversion=1.2)
original_frontier_mu, original_frontier_sigma = mkw_original.min_var_frontier()

vol2 = pd.Series(data=np.diag(bl.sigma_bl.values)**0.5, index=asset_list, name='Vol')
mkw_bl = Markowitz(bl.mu_bl['Expected Returns'], vol2, corr, 0.01, risk_aversion=1.2)
bl_frontier_mu, bl_frontier_sigma = mkw_bl.min_var_frontier()

# ===== Chart =====
plt.figure(figsize=(10, 6))
# assets
plt.scatter(vol, bl.mu_best_guess['Best Guess of mu'], label='Original Assets', color='red', marker='o',
            edgecolor='black')
plt.scatter(vol2, bl.mu_bl['Expected Returns'], label='Black-Litterman Views', color='green', marker='p',
            edgecolor='black')

# risk-free
plt.scatter(0, 0.01, label='Risk-Free')

# Optimal risky portfolio
plt.scatter(mkw_original.sigma_p, mkw_original.mu_p, label='Original Optimal', color='firebrick',
            marker='X', s=50, zorder=-1)
plt.scatter(mkw_bl.sigma_p, mkw_bl.mu_p, label='Black-Litterman Optimal', color='darkgreen',
            marker='X',  s=50, zorder=-1)

# Minimal Variance Portfolio
# plt.scatter(self.sigma_mv, self.mu_mv, label='Min Variance')

# Minimal variance frontier
plt.plot(original_frontier_sigma, original_frontier_mu, marker=None, color='red',
         label='Original Min Variance Frontier')
plt.plot(bl_frontier_sigma, bl_frontier_mu, marker=None, color='green',
         label='Black-Litterman Min Variance Frontier')

# Capital allocation line
max_sigma = vol.max() + 0.05
x_values = [0, max_sigma]
y_values = [0.01, 0.01 + mkw_original.sharpe_p * max_sigma]
plt.plot(x_values, y_values, marker=None, color='red', label='Original Capital Allocation Line', linestyle='--')
y_values = [0.01, 0.01 + mkw_bl.sharpe_p * max_sigma]
plt.plot(x_values, y_values, marker=None, color='green', label='Black-Litterman Capital Allocation Line', linestyle='--')

# # Investor's portfolio
# plt.scatter(self.sigma_c, self.mu_c, label="Investor's Portfolio", color='purple')
#
# # Indiference Curve
# max_sigma = self.sigma_p + 0.1
# x_values = np.arange(0, max_sigma, max_sigma / 100)
# y_values = self.certain_equivalent + 0.5 * self.risk_aversion * (x_values ** 2)
# plt.plot(x_values, y_values, marker=None, color='purple', zorder=-1, label='Indiference Curve')
#
# # legend
plt.legend(loc='upper left')

# adjustments
plt.xlim((0, vol.max() + 0.03))
plt.ylim((0.005, bl.mu_bl['Expected Returns'].max() + 0.01))
plt.xlabel('Risk')
plt.ylabel('Return')
plt.tight_layout()

# Save as picture
plt.savefig(file_path + f'/figures/BL Min variance frontier comparison.pdf', pad_inches=0)

plt.show()


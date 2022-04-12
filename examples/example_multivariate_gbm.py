from quantfin.simulation import MultivariateGBM
from quantfin.statistics import corr2cov
import matplotlib.pyplot as plt

mu = [0.3, 0.1]
std = [0.2, 0.1]

corr = [[1, 0.95],
        [0.95, 1]]

cov = corr2cov(corr, std)

mgbm = MultivariateGBM(T=1, n=252, mu=mu, sigma=cov)
mgbm.simulated_trajectories.plot(legend=None, linewidth=1, alpha=0.5, title='Multivariate GBM')
plt.show()

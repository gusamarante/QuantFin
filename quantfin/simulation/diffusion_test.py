import matplotlib.pyplot as plt
from quantfin.simulation import Diffusion
from scipy.stats import lognorm
import numpy as np
import pandas as pd

# ===== Brownian Motion =====
diff = Diffusion(T=1, n=100, k=100)

diff.brownian_motion.plot(legend=None)
diff.theoretical_mean.plot(legend=None, color='black', linewidth=3)
(diff.theoretical_mean + 1.96 * diff.theoretical_std).plot(legend=None, color='black', linewidth=3, style='--')
(diff.theoretical_mean - 1.96 * diff.theoretical_std).plot(legend=None, color='black', linewidth=3, style='--')
plt.show()

print(diff.brownian_motion.iloc[-1].mean())
print(diff.brownian_motion.iloc[-1].std())

plt.hist(diff.brownian_motion.iloc[-1], bins=100)
plt.show()


# ===== Random Walk with Drift =====
diff = Diffusion(T=1, n=100, k=100, initial_price=20, process_type='rwwd', drift=0.5, diffusion=0.5)

diff.simulated_trajectories.plot(legend=None)
diff.theoretical_mean.plot(legend=None, color='black', linewidth=3)
(diff.theoretical_mean + 1.96 * diff.theoretical_std).plot(legend=None, color='black', linewidth=3, style='--')
(diff.theoretical_mean - 1.96 * diff.theoretical_std).plot(legend=None, color='black', linewidth=3, style='--')
plt.show()

print(diff.simulated_trajectories.iloc[-1].mean())
print(diff.simulated_trajectories.iloc[-1].std())

plt.hist(diff.simulated_trajectories.iloc[-1], bins=100)
plt.show()

# ===== Geometric borwnian motion / log-normal random walk / price process =====
diff = Diffusion(T=1, n=100, k=1000, initial_price=30, process_type='gbm', drift=0.2, diffusion=0.2)

lower_ci = [lognorm.ppf(0.025, 0.2 * srdt, loc=(0.2-0.5 * 0.2 ** 2) * srdt**2, scale=media) for srdt, media in zip(np.sqrt(diff.time_array), diff.theoretical_mean)]
upper_ci = [lognorm.ppf(0.975, 0.2 * srdt, loc=(0.2-0.5 * 0.2 ** 2) * srdt**2, scale=media) for srdt, media in zip(np.sqrt(diff.time_array), diff.theoretical_mean)]

lower_ci = pd.Series(index=diff.simulated_trajectories.index, data=lower_ci, name='Lower CI').fillna(30)
upper_ci = pd.Series(index=diff.simulated_trajectories.index, data=upper_ci, name='Upper CI').fillna(30)

diff.simulated_trajectories.plot(legend=None)
diff.theoretical_mean.plot(legend=None, color='black', linewidth=3)
lower_ci.plot(legend=None, color='black', linewidth=3, linestyle='--')
upper_ci.plot(legend=None, color='black', linewidth=3, linestyle='--')
plt.show()

print(diff.simulated_trajectories.iloc[-1].mean())
print(diff.simulated_trajectories.iloc[-1].std())

plt.hist(diff.simulated_trajectories.iloc[-1], bins=100)
plt.show()

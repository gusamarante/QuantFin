import matplotlib.pyplot as plt
from quantfin.simulation import Diffusion
from scipy.stats import lognorm
import numpy as np
import pandas as pd

# ===== Brownian Motion =====
diff = Diffusion(T=1, n=1000, k=10)

diff.brownian_motion.plot(legend=None)
diff.theoretical_mean.plot(legend=None, color='black', linewidth=3)
diff.ci_lower.plot(legend=None, color='black', linewidth=3, linestyle='--')
diff.ci_upper.plot(legend=None, color='black', linewidth=3, linestyle='--')
plt.show()


# ===== Random Walk with Drift =====
diff = Diffusion(T=1, n=1000, k=10, initial_price=20, process_type='rwwd', drift=0.5, diffusion=0.5)

diff.simulated_trajectories.plot(legend=None)
diff.theoretical_mean.plot(legend=None, color='black', linewidth=3)
diff.ci_lower.plot(legend=None, color='black', linewidth=3, linestyle='--')
diff.ci_upper.plot(legend=None, color='black', linewidth=3, linestyle='--')
plt.show()


# ===== Geometric borwnian motion / log-normal random walk / price process =====
diff = Diffusion(T=1, n=1000, k=10, initial_price=30, process_type='gbm', drift=0.2, diffusion=0.2)

diff.simulated_trajectories.plot(legend=None)
diff.theoretical_mean.plot(legend=None, color='black', linewidth=3)
diff.ci_lower.plot(legend=None, color='black', linewidth=3, linestyle='--')
diff.ci_upper.plot(legend=None, color='black', linewidth=3, linestyle='--')
plt.show()

# ===== Mean Reversion Process =====
diff = Diffusion(T=4, n=1000, k=10, initial_price=2.5, process_type='ou', drift=0.9, diffusion=0.1, mean=2)

diff.simulated_trajectories.plot(legend=None)
diff.theoretical_mean.plot(legend=None, color='black', linewidth=3)
diff.ci_lower.plot(legend=None, color='black', linewidth=3, linestyle='--')
diff.ci_upper.plot(legend=None, color='black', linewidth=3, linestyle='--')
plt.show()


from quantfin.simulation import Diffusion
import matplotlib.pyplot as plt

# Brownian Motion
diff = Diffusion(T=1, n=1000, k=20, process_type='bm')
diff.simulated_trajectories.plot(legend=None, linewidth=1, alpha=0.5, title='Brownian Motion')
diff.theoretical_mean.plot(legend=None, linewidth=2, color='black')
diff.ci_lower.plot(legend=None, linewidth=2, color='black', linestyle='--')
diff.ci_upper.plot(legend=None, linewidth=2, color='black', linestyle='--')
plt.show()

# Random Walk with Drift
diff = Diffusion(T=1, n=1000, k=20, process_type='rwwd', drift=5, diffusion=2, initial_price=10)
diff.simulated_trajectories.plot(legend=None, linewidth=1, alpha=0.5, title='Random Walk with Drift')
diff.theoretical_mean.plot(legend=None, linewidth=2, color='black')
diff.ci_lower.plot(legend=None, linewidth=2, color='black', linestyle='--')
diff.ci_upper.plot(legend=None, linewidth=2, color='black', linestyle='--')
plt.show()

# Geometric Brownian Motion
diff = Diffusion(T=1, n=1000, k=20, process_type='gbm', drift=0.2, diffusion=0.3, initial_price=10)
diff.simulated_trajectories.plot(legend=None, linewidth=1, alpha=0.5, title='Geometric Brownian Motion')
diff.theoretical_mean.plot(legend=None, linewidth=2, color='black')
diff.ci_lower.plot(legend=None, linewidth=2, color='black', linestyle='--')
diff.ci_upper.plot(legend=None, linewidth=2, color='black', linestyle='--')
plt.show()

# Ornstein-Uhlenbeck Process
diff = Diffusion(T=1, n=1000, k=20, process_type='ou', drift=8, mean=10, diffusion=1, initial_price=14)
diff.simulated_trajectories.plot(legend=None, linewidth=1, alpha=0.5, title='Ornstein-Uhlenbeck Process')
diff.theoretical_mean.plot(legend=None, linewidth=2, color='black')
diff.ci_lower.plot(legend=None, linewidth=2, color='black', linestyle='--')
diff.ci_upper.plot(legend=None, linewidth=2, color='black', linestyle='--')
plt.show()

# Jump-GBM
diff = Diffusion(T=1, n=1000, k=10, process_type='jump', drift=0.3, diffusion=0.2, initial_price=14,
                 jump_freq=10, jump_mean=0.1, jump_std=0.01)
diff.simulated_trajectories.plot(legend=None, linewidth=1, alpha=0.5, title='Jump Diffusion Process')
diff.theoretical_mean.plot(legend=None, linewidth=2, color='black')
plt.show()

import matplotlib.pyplot as plt
from quantfin.simulation import Diffusion

# Brownian Motion
diff = Diffusion(T=1, n=100, k=1000)

diff.brownian_motion.plot(legend=None)
plt.show()

print(diff.brownian_motion.iloc[-1].mean())
print(diff.brownian_motion.iloc[-1].std())

plt.hist(diff.brownian_motion.iloc[-1], bins=100)
plt.show()


# Random Walk with Drift
diff = Diffusion(T=1, n=1000, k=1000, initial_price=20, process_type='rwwd', drift=5, diffusion=0.5)

diff.simulated_trajectories.plot(legend=None)
plt.show()

print(diff.simulated_trajectories.iloc[-1].mean())
print(diff.simulated_trajectories.iloc[-1].std())

plt.hist(diff.simulated_trajectories.iloc[-1], bins=100)
plt.show()

# geometric borwnian motion / log-normal random walk / price process
diff = Diffusion(T=1, n=100, k=10000, initial_price=20, process_type='gbm', drift=0.1, diffusion=0.3)

diff.simulated_trajectories.plot(legend=None)
plt.show()

print(diff.simulated_trajectories.iloc[-1].mean())
print(diff.simulated_trajectories.iloc[-1].std())

plt.hist(diff.simulated_trajectories.iloc[-1], bins=100)
plt.show()

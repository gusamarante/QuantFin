from quantfin.simulation import BrownianMotion
import matplotlib.pyplot as plt
import pandas as pd

show_charts = False
save_path = '/Users/gustavoamarante/Dropbox/Aulas/QuantFin/figures/'

# Simulation of the scaled random walk for different values of n
df = pd.DataFrame()

bm = BrownianMotion(T=2, n=10, k=1, random_seed=123)
df = pd.concat([df, bm.simulated_trajectories.iloc[:, 0].rename('n=10')], axis=1)

bm = BrownianMotion(T=2, n=100, k=1, random_seed=465)
df = pd.concat([df, bm.simulated_trajectories.iloc[:, 0].rename('n=100')], axis=1)

bm = BrownianMotion(T=2, n=1000, k=1, random_seed=789)
df = pd.concat([df, bm.simulated_trajectories.iloc[:, 0].rename('n=1000')], axis=1)

df = df.interpolate(method='linear', limit_area='inside')
df.plot(figsize=(5 * 1.61, 5))
plt.axhline(0, color='black', linewidth=1)
plt.xlabel('Time')
plt.ylabel('$W_{t}^{(n)}$', rotation=0)
plt.tight_layout()
plt.savefig(save_path + 'scaled random walk by n.pdf')

if show_charts:
    plt.show()

plt.close()

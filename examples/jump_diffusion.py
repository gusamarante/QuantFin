from quantfin.simulation import Diffusion
import matplotlib.pyplot as plt
import pandas as pd

diff = Diffusion(T=1,
                 n=100,
                 k=5000,
                 initial_price=100,
                 process_type='jump',
                 drift=0.02,
                 diffusion=0.1,
                 jump_mean=0.3,
                 jump_std=0.01,
                 jump_freq=2
                 )

df_plot = pd.concat([diff.simulated_trajectories.mean(axis=1).rename('Sample Mean'),
                     diff.theoretical_mean], axis=1)
df_plot.plot()
plt.show()

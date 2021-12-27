from quantfin.simulation import Diffusion
import matplotlib.pyplot as plt

diff = Diffusion(T=1,
                 n=100,
                 k=1000,
                 initial_price=100,
                 process_type='jump',
                 drift=0.02,
                 diffusion=0.1,
                 jump_size=0.1,
                 jump_freq=2
                 )

diff.simulated_trajectories.iloc[-1].hist(legend=None)
plt.show()

import numpy as np
import pandas as pd

# TODO Random walk with drift
# TODO Log-normal random walk (Geometric BM)
# TODO Mean-reverting random walk (Ornstein-Uhlenbeck)
# TODO Mean-reverting Square root process
# TODO General diffusion process class that takes mu(x,t), sigma(x,t) and y=f(x,t)
# TODO correlated brownian motions (product rule for Ito, https://en.wikipedia.org/wiki/Itô%27s_lemma#Product_rule_for_Itô_processes)


class Diffusion(object):
    supported_process_type = ['bm']  # Simple Brownian Motion

    def __init__(self, T=1, n=100, k=1, process_type='bm', drift=None, diffusion=None,
                 random_seed=None):
        # TODO Documentation
        assert process_type in self.supported_process_type, "Process not yet implemented"

        self.n = n  # Number of subintervals of the simulation
        self.T = T  # Time in years
        self.k = k  # Number of simulated trajectories
        self.delta_t = T / n  # Time step
        self.brownian_motion = self._get_brownian_motion(T, n, k, random_seed)

        if process_type == 'bm':
            self.simulated_trajectories = self.brownian_motion

    def _get_brownian_motion(self, T, n, k, random_seed):
        """
        This simulates a brownian motion using a normal distribution, where
        each step is draw from a N(0,1)*sqrt(delta_t)
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        shocks = np.random.normal(0, 1, (n, k)) * np.sqrt(T / n)
        brownian_motion = shocks.cumsum(axis=0)
        brownian_motion = np.vstack([np.zeros((1, k)), brownian_motion])

        index_array = list(np.arange(0, self.T + self.delta_t, self.delta_t))
        column_array = ['Brownian Motion ' + str(i+1) for i in range(k)]
        brownian_motion = pd.DataFrame(data=brownian_motion,
                                       index=index_array,
                                       columns=column_array)
        return brownian_motion

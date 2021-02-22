import numpy as np
import pandas as pd

# TODO Mean-reverting random walk (Ornstein-Uhlenbeck)
# TODO Mean-reverting Square root process
# TODO General diffusion process class that takes mu(x,t), sigma(x,t) and y=f(x,t)
# TODO correlated brownian motions (product rule for Ito,
#      https://en.wikipedia.org/wiki/Itô%27s_lemma#Product_rule_for_Itô_processes)


class Diffusion(object):
    supported_process_type = ['bm',  # Simple Brownian Motion (dW)
                              'rwwd',  # Random Walk with drift (dX = mu * dt + sigma * dW)
                              'gbm']  # Geometric BM/Log-normal random walk (dS = mu * S * dt + sigma * S * dW)

    def __init__(self, T=1, n=100, k=1, initial_price=0, process_type='bm', drift=None, diffusion=None,
                 random_seed=None):
        # TODO Documentation
        assert process_type in self.supported_process_type, "Process not yet implemented"

        self.n = n  # Number of subintervals of the simulation
        self.T = T  # Time in years
        self.k = k  # Number of simulated trajectories
        self.initial_price = initial_price
        self.delta_t = T / n  # Time step
        self.brownian_motion = self._get_brownian_motion(random_seed)

        if process_type == 'bm':
            self.simulated_trajectories = self.brownian_motion
        elif process_type == 'rwwd':
            self.simulated_trajectories = self._get_random_walk_with_drift(drift, diffusion)
        elif process_type == 'gbm':
            self.simulated_trajectories = self._get_geometric_brownian_motion(drift, diffusion)

    def _get_brownian_motion(self, random_seed):
        """
        This simulates a brownian motion using a normal distribution, where
        each step is draw from a N(0,1) * sqrt(delta_t)
        """

        if random_seed is not None:
            np.random.seed(random_seed)

        shocks = np.random.normal(0, 1, (self.n, self.k)) * np.sqrt(self.T / self.n)
        brownian_motion = shocks.cumsum(axis=0)
        brownian_motion = np.vstack([np.zeros((1, self.k)), brownian_motion])  # Adds the 'zero' starting point

        # Organize the result in a pandas DataFrame
        index_array = list(np.arange(0, self.T + self.delta_t, self.delta_t))
        column_array = ['Brownian Motion ' + str(i+1) for i in range(self.k)]
        brownian_motion = pd.DataFrame(data=brownian_motion,
                                       index=index_array,
                                       columns=column_array)
        return brownian_motion

    def _get_random_walk_with_drift(self, drift, diffusion):
        cond = (drift is not None) and (diffusion is not None)
        assert cond, "'drift' and 'diffusion' must not be None in a random walk with drift"

        time_index = np.arange(0, self.T + self.delta_t, self.delta_t)
        rwwd = self.initial_price + time_index * drift  # initial price + trend
        rwwd = np.reshape(rwwd, (self.n + 1, 1))  # reshape array for broadcasting
        rwwd = rwwd + diffusion * self.brownian_motion.values  # add the random part

        column_array = ['Random Walk ' + str(i + 1) for i in range(self.k)]
        rwwd = pd.DataFrame(data=rwwd,
                            index=list(time_index),
                            columns=column_array)

        return rwwd

    def _get_geometric_brownian_motion(self, drift, diffusion):
        cond = (drift is not None) and (diffusion is not None)
        assert cond, "'drift' and 'diffusion' must not be None in a geometric brownian motion"

        time_index = np.arange(0, self.T + self.delta_t, self.delta_t)
        gbm = (drift - (diffusion**2)/2) * time_index  # exponent
        gbm = np.reshape(gbm, (self.n + 1, 1))  # Reshape array for broadcasting
        gbm = gbm + diffusion * self.brownian_motion.values
        gbm = self.initial_price * np.exp(gbm)

        column_array = ['Geometric ' + str(i + 1) for i in range(self.k)]
        gbm = pd.DataFrame(data=gbm,
                           index=list(time_index),
                           columns=column_array)

        return gbm

import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm


class BrownianMotion(object):

    def __init__(self, T, n, k=1, random_seed=None):
        """
        Simulates the scaled random walk that leads to the brownian motion
        :param T: number of years in the simulation
        :param n: number of steps in the simulated trajectories
        :param k: number of trajectories to simulate
        :param random_seed: random seed for numpy RNG
        """

        self.T = T
        self.n = n

        if random_seed is not None:
            np.random.seed(random_seed)

        time_index = np.arange(n * T + 1) / n

        # Scaled Random Walk
        omega = np.random.uniform(size=(T * n, k))  # "flip the coins"
        X = (omega >= 0.5) * 1 + (omega < 0.5) * (-1)  # get the increments
        M = X.cumsum(axis=0)  # Sum the increments (integration)
        M = (1 / np.sqrt(n)) * M  # Scale the process
        M = np.vstack([np.zeros((1, k)), M])  # add a zero as a starting point

        column_names = [f'Brownian Motion {i+1}' for i in range(k)]
        self.simulated_trajectories = pd.DataFrame(index=time_index,
                                                   data=M,
                                                   columns=column_names)


class Diffusion(object):

    supported_process_type = ['bm', 'rwwd', 'gbm', 'ou', 'jump']

    def __init__(self, T=1, n=100, k=1, initial_price=0, process_type='bm', drift=None, diffusion=None, mean=None,
                 random_seed=None, conf=0.95, jump_size=None, jump_freq=None):
        """
        Simulates diffusion processes.
        :param T: Number of years to simulate
        :param n: Number of steps in the series
        :param k: Number of trajectories to simulate
        :param initial_price: Starting point of the trajectories
        :param process_type: supported methods are described below and listed in 'supported_process_type'
        :param drift: Coeficient. Use varies depending on the 'process_type'.
        :param diffusion: Coeficient. Use varies depending on the 'process_type'.
        :param mean: Coeficient. Use varies depending on the 'process_type'.
        :param random_seed: random seed for the numpy RNG.
        :param conf: confidence for the confidence intervals.
        :param jump_size: Jump size parameter (only for 'jump' process)
        :param jump_freq: Average number of jumps per year (only for 'jump' process)
        :param process_type: Type of diffusion process to simulate.

        - process_type='bm': simple brownian motion. In this case, the terms 'drift', 'diffusion' and
                             'mean' are not used.
                                dW

        - process_type='rwwd': Random walk with drift. In this case the 'mean' term is not used and the
                               terms 'drift' and 'diffusion' are used as follows:
                                  dS = drift * dt + diffusion * dW

        - process_type='gbm': Geometric brownian motion / Log-normal random walk / Price process. In this
                              case the 'mean' term is not used and the terms 'drift' and 'diffusion' are
                              used as follows:
                                 dS = drift * S * dt + diffusion * S * dW

        - process_type='ou': Ornstein-Uhlenbeck / Mean-reverting process. The terms 'drift', 'diffusion'
                             and 'mean' are used as follows:
                                dS = drift * (mean - x) * dt + diffusion * dW

        - process_type='jump': Geometric brownian motion process with jumps. The terms 'drift', 'diffusion',
                               'jump_size' and 'jump_freq' are used as follows:
                                dS = drift * S * dt + diffusion * S * dW + S * dJ(jump_size, jump_freq)
        """

        assert process_type in self.supported_process_type, f"Process {process_type} not yet implemented"

        self.n = n  # Number of subintervals of the simulation
        self.T = T  # Time in years
        self.k = k  # Number of simulated trajectories
        self.initial_price = initial_price
        self.delta_t = T / n  # Time step
        self.time_array = np.arange(0, T + self.delta_t, self.delta_t)
        self.brownian_motion = self._get_brownian_motion(random_seed)
        self.conf = conf

        if process_type == 'bm':

            self.simulated_trajectories = self.brownian_motion

            self.theoretical_mean = pd.Series(name='Theoretical Mean',
                                              index=self.time_array,
                                              data=0)

            self.theoretical_std = pd.Series(name='Theoretical Std',
                                             index=self.time_array,
                                             data=np.sqrt(self.time_array))

            self.ci_lower = self.theoretical_mean - self.theoretical_std * norm.ppf((1 + conf) / 2)
            self.ci_upper = self.theoretical_mean + self.theoretical_std * norm.ppf((1 + conf) / 2)

        elif process_type == 'rwwd':

            self.simulated_trajectories = self._get_random_walk_with_drift(drift, diffusion)

            self.theoretical_mean = pd.Series(name='Theoretical Mean',
                                              index=self.time_array,
                                              data=self.initial_price + drift * self.time_array)

            self.theoretical_std = pd.Series(name='Theoretical Std',
                                             index=self.time_array,
                                             data=diffusion * np.sqrt(self.time_array))

            self.ci_lower = self.theoretical_mean - self.theoretical_std * norm.ppf((1 + conf) / 2)
            self.ci_upper = self.theoretical_mean + self.theoretical_std * norm.ppf((1 + conf) / 2)

        elif process_type == 'gbm':

            self.simulated_trajectories = self._get_geometric_brownian_motion(drift, diffusion)

            self.theoretical_mean = pd.Series(name='Theoretical Mean',
                                              index=self.time_array,
                                              data=self.initial_price * np.exp(drift * self.time_array))

            # Reminder: the distribution of the gbm is log-normal, and although we can
            # compute the std for the lognormal distribution, the confidence interval must
            # be built based on assymetric valeus, in order to get something accurate and
            # inside the domain of the process.
            variance = (self.initial_price ** 2) * np.exp(2 * drift * self.time_array) * (np.exp(self.time_array *
                                                                                                 (diffusion ** 2)) - 1)
            self.theoretical_std = pd.Series(name='Theoretical Std',
                                             index=self.time_array,
                                             data=np.sqrt(variance))

            lower_ci = [lognorm.ppf((1-conf)/2, diffusion * srdt,
                                    loc=(drift - 0.5 * diffusion ** 2) * srdt ** 2,
                                    scale=media)
                        for srdt, media in zip(np.sqrt(self.time_array), self.theoretical_mean)]

            upper_ci = [lognorm.ppf((1 + conf) / 2, diffusion * srdt,
                                    loc=(drift - 0.5 * diffusion ** 2) * srdt ** 2,
                                    scale=media)
                        for srdt, media in zip(np.sqrt(self.time_array), self.theoretical_mean)]

            self.ci_lower = pd.Series(index=self.time_array, data=lower_ci, name='Lower CI').fillna(initial_price)
            self.ci_upper = pd.Series(index=self.time_array, data=upper_ci, name='Upper CI').fillna(initial_price)

        elif process_type == 'ou':
            self.simulated_trajectories = self._get_ornstein_uhlenbeck(drift, diffusion, mean)

            self.theoretical_mean = pd.Series(name='Theoretical Mean',
                                              index=self.time_array,
                                              data=mean + (initial_price - mean)*np.exp(-drift*self.time_array))

            self.theoretical_std = pd.Series(name='Theoretical Std',
                                             index=self.time_array,
                                             data=diffusion * np.sqrt((1-np.exp(-2*drift*self.time_array))/(2*drift)))

            self.ci_lower = self.theoretical_mean - self.theoretical_std * norm.ppf((1 + conf) / 2)
            self.ci_upper = self.theoretical_mean + self.theoretical_std * norm.ppf((1 + conf) / 2)

        elif process_type == 'jump':
            self.simulated_trajectories = self._get_jump_gbm(drift, diffusion, jump_size, jump_freq)

            # TODO I do not know if the theoretical moments of the Jump GBM can be computed.
            self.theoretical_mean = None
            self.theoretical_std = None
            self.ci_lower = None
            self.ci_upper = None

        else:
            raise NotImplementedError("Process not yet implemented")

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
        column_array = ['Brownian Motion ' + str(i+1) for i in range(self.k)]
        brownian_motion = pd.DataFrame(data=brownian_motion,
                                       index=list(self.time_array),
                                       columns=column_array)
        return brownian_motion

    def _get_random_walk_with_drift(self, drift, diffusion):
        cond = (drift is not None) and (diffusion is not None)
        assert cond, "'drift' and 'diffusion' must not be None in a random walk with drift"

        rwwd = self.initial_price + self.time_array * drift  # initial price + trend
        rwwd = np.reshape(rwwd, (self.n + 1, 1))  # reshape array for broadcasting
        rwwd = rwwd + diffusion * self.brownian_motion.values  # add the random part

        column_array = ['Random Walk ' + str(i + 1) for i in range(self.k)]
        rwwd = pd.DataFrame(data=rwwd,
                            index=list(self.time_array),
                            columns=column_array)

        return rwwd

    def _get_geometric_brownian_motion(self, drift, diffusion):
        cond = (drift is not None) and (diffusion is not None)
        assert cond, "'drift' and 'diffusion' must not be None in a geometric brownian motion"

        gbm = (drift - (diffusion ** 2) / 2) * self.time_array  # exponent
        gbm = np.reshape(gbm, (self.n + 1, 1))  # Reshape array for broadcasting
        gbm = gbm + diffusion * self.brownian_motion.values
        gbm = self.initial_price * np.exp(gbm)

        column_array = ['Geometric ' + str(i + 1) for i in range(self.k)]
        gbm = pd.DataFrame(data=gbm,
                           index=list(self.time_array),
                           columns=column_array)

        return gbm

    def _get_ornstein_uhlenbeck(self, drift, diffusion, mean):
        cond = (drift is not None) and (diffusion is not None)
        assert cond, "'drift', 'diffusion' and 'mean' must not be None in a Ornstein-Uhlenbeck process"

        ou_mean = mean + (self.initial_price - mean) * np.exp(-drift*self.time_array)
        ou_mean = np.reshape(ou_mean, (self.n + 1, 1))  # reshape array for broadcasting
        # ou_diff = diffusion*np.sqrt((1-np.exp(-2*drift*self.time_array))/(2*drift))
        dWt = self.brownian_motion.diff(1).fillna(0).values
        integrand = np.reshape(np.exp(drift * self.time_array), (self.n + 1, 1)) * dWt
        integral = integrand.cumsum(axis=0)
        ou_diff = np.reshape(np.exp(-drift*self.time_array), (self.n + 1, 1)) * diffusion * integral
        ou = ou_mean + ou_diff

        column_array = ['Ornstein-Uhlenbeck ' + str(i + 1) for i in range(self.k)]

        ou = pd.DataFrame(data=ou,
                          index=list(self.time_array),
                          columns=column_array)

        return ou

    def _get_jump_gbm(self, drift, diffusion, jump_size, jump_freq):
        cond = (drift is not None) and (diffusion is not None) and (jump_size is not None) and (jump_freq is not None)
        msg = "'drift', 'diffusion' and 'extra_param' must not be None in a geometric brownian motion with jumps. " \
              "Please check documentation."
        assert cond, msg

        # generate the Geometric brownian motion
        gbm = self._get_geometric_brownian_motion(drift, diffusion)

        # simulate the jumps
        lamb = jump_freq * self.delta_t
        jumps = np.random.poisson(lam=lamb, size=(self.n + 1, self.k))
        jumps = (1 + jump_size) ** jumps
        jumps = jumps.cumprod(axis=0)

        jump_gbm = gbm.values * jumps

        column_array = ['Jump ' + str(i + 1) for i in range(self.k)]
        jump_gbm = pd.DataFrame(data=jump_gbm,
                                index=list(self.time_array),
                                columns=column_array)

        return jump_gbm


class MultivariateGBM(object):

    def __init__(self, T, n, mu, sigma, initial_price=None, random_seed=None):
        # TODO Documentation (mu is not drift, sigma is not covariance, they are of the log)

        dt = T / n
        mu = np.array(mu)
        sigma = np.array(sigma)

        self.time_array = np.linspace(0, T, n+1)

        if random_seed is not None:
            np.random.seed(random_seed)

        n_assets = self._get_n_assets(mu, sigma, initial_price)

        Z = np.random.normal(size=(n, n_assets))
        A = np.linalg.cholesky(sigma)

        factor = np.exp((mu - 0.5 * np.diag(sigma)) * dt + np.sqrt(dt) * (A @ Z.T).T).cumprod(axis=0)

        if initial_price is None:
            price = 100 * np.vstack([np.ones((1, n_assets)), factor])
        else:
            price = np.vstack([np.array(initial_price), np.array(initial_price) * factor])

        self.simulated_trajectories = pd.DataFrame(data=price,
                                                   index=self.time_array,
                                                   columns=[f'MGBM {i + 1}' for i in range(n_assets)])

    @staticmethod
    def _get_n_assets(mu, sigma, initial_price):
        shape_drift = mu.shape[0]
        shape_sigma1, shape_sigma2 = sigma.shape

        cond1 = shape_drift == shape_sigma1
        cond2 = shape_drift == shape_sigma2
        cond3 = shape_sigma1 == shape_sigma2

        if initial_price is not None:
            cond4 = np.array(initial_price).shape[0] == shape_drift
        else:
            cond4 = True

        if not (cond1 and cond2 and cond3 and cond4):
            raise ValueError("mismatch in input shapes")

        return shape_sigma1

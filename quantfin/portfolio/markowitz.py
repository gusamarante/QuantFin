from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from numpy.linalg import inv
import pandas as pd
import numpy as np


class Markowitz(object):
    # TODO Chart Functionality
    # TODO Borrowing Rate
    # TODO Documentation
    # TODO Example Notebook (include shortselling)

    def __init__(self, mu, sigma, corr, rf, risk_aversion=None, short_sell=True):
        """
        Receives informations about the set of assets available for allocation and computes the following
        measures as atributes:
            - cov: Covariance matrix of the assets
            - mu_p: expected return of the optimal risky portfolio (tangency portfolio)
            - sigma_p: risk of the optimal risky portfolio (tangency portfolio)
            - risky_weights: pandas series of the weights of each risky asset on the optimal risk portfolio
            - sharpe_p: expected sharpe ratio of the optimal risky portfolio.
            - mu_mv: expected return of the minimal variance portfolio
            - sigma_mv: risk of the minimal variance portfolio
            - mv_weights: pandas series of the weights of each risky asset on the minimal variance portfolio
            - sharpe_mv: expected sharpe ratio of the minimal variance portfolio.
            - weight_p: weight of the risk porfolio on the investor's portfolio. The remaining 1-weight_p is
                        allocated on the risk-free asset
            - complete_weights: weights of the risky and risk-free assets on the investor's portfolio.
            - mu_c: expected return of the investor's portfolio
            - sigma_c: risk of the investor's portfolio
            - certain_equivalent: the hypothetical risk-free return that would make the investor indiferent
                                  to its risky portfolioo of choice (portfolio C)

        Computations involving the investor's preference use the following utility function:
            U = mu_c - 0.5 * risk_aversion * (sigma_c**2)

        The class also has support for plotting the classical risk-return plane.

        :param mu: pandas series of expected returns where the index contains the names of the assets.
        :param sigma: pandas series of risk where the index contains the names of the assets
                      (index must be the same as 'mu')
        :param corr: pandas DataFrame of correlations (index and columns must contain the same index)
        :param rf: float, risk-free rate
        :param risk_aversion: coefficient of risk aversion of the investor's utility function.
        :param short_sell: If True, short-selling is allowed. If False, weights on risky-assets are
                           constrained to be between 0 and 1
        """
        # TODO Assert data indexes match and organize the indexes

        # Save inputs as attributes
        self.mu = mu
        self.sigma = sigma
        self.corr = corr
        self.rf = rf
        self.risk_aversion = risk_aversion
        self.short_selling = short_sell

        # Compute atributes
        self.n_assets = self._n_assets()
        self.cov = self._get_cov_matrix()

        # Get the optimal risky porfolio
        self.mu_p, self.sigma_p, self.risky_weights, self.sharpe_p = self._get_optimal_risky_portfolio()

        # get the minimal variance portfolio
        self.mu_mv, self.sigma_mv, self.mv_weights, self.sharpe_mv = self._get_minimal_variance_portfolio()

        # Get the investor's portfolio and build the complete set of weights
        self.weight_p, self.complete_weights, self.mu_c, self.sigma_c, self.certain_equivalent \
            = self._investor_allocation()

    def plot(self):
        # TODO Make elements optional
        # TODO Add save_path for the figure

        # assets
        plt.scatter(self.sigma, self.mu, label='Assets')

        # risk-free
        plt.scatter(0, self.rf, label='Risk-Free')

        # Optimal risky portfolio
        plt.scatter(self.sigma_p, self.mu_p, label='Optimal Risk')

        # Minimal Variance Portfolio
        plt.scatter(self.sigma_mv, self.mu_mv, label='Min Variance')

        # Minimal variance frontier
        if not self.n_assets == 1:
            mu_mv, sigma_mv = self._min_var_frontier()
            plt.plot(sigma_mv, mu_mv, marker=None, color='black', zorder=-1, label='Min Variance Frontier')

        # Capital allocation line
        max_sigma = self.sigma.max() + 0.05
        x_values = [0, max_sigma]
        y_values = [self.rf, self.rf + self.sharpe_p * max_sigma]
        plt.plot(x_values, y_values, marker=None, color='green', zorder=-1, label='Capital Allocation Line')

        # Investor's portfolio
        plt.scatter(self.sigma_c, self.mu_c, label="Investor's Portfolio", color='purple')

        # Indiference Curve
        max_sigma = self.sigma_p + 0.1
        x_values = np.arange(0, max_sigma, max_sigma / 100)
        y_values = self.certain_equivalent + 0.5 * self.risk_aversion * (x_values ** 2)
        plt.plot(x_values, y_values, marker=None, color='purple', zorder=-1, label='Indiference Curve')

        # legend
        plt.legend(loc='best')

        # adjustments
        plt.xlim((0, self.sigma.max() + 0.1))
        plt.xlabel('Risk')
        plt.ylabel('Return')
        plt.tight_layout()
        plt.show()

    def _n_assets(self):
        """
        Makes sure that all inputs have the correct shape and returns the number of assets
        """
        shape_mu = self.mu.shape
        shape_sigma = self.sigma.shape
        shape_corr = self.corr.shape

        max_shape = max(shape_mu[0], shape_sigma[0], shape_corr[0], shape_corr[1])
        min_shape = min(shape_mu[0], shape_sigma[0], shape_corr[0], shape_corr[1])

        if max_shape == min_shape:
            return max_shape
        else:
            raise AssertionError('Mismatching dimensions of inputs')

    def _get_optimal_risky_portfolio(self):

        if self.n_assets == 1:  # one risky asset (analytical)
            mu_p = self.mu.iloc[0]
            sigma_p = self.sigma.iloc[0]
            sharpe_p = (mu_p - self.rf)/sigma_p
            weights = pd.Series(data={self.mu.index[0]: 1},
                                name='Risky Weights')

        else:  # multiple risky assets (optimization)
            # TODO Optimization

            # define the objective function (notice the sign change on the return value)
            def sharpe(x):
                return -self._sharpe(x, self.mu.values, self.cov.values, self.rf, self.n_assets)

            # budget constraint
            constraints = ({'type': 'eq',
                           'fun': lambda w: w.sum() - 1})

            # Create bounds for the weights if short-selling is restricted
            if self.short_selling:
                bounds = None
            else:
                bounds = Bounds(np.zeros(self.n_assets), np.ones(self.n_assets))

            # initial guess
            w0 = np.zeros(self.n_assets)
            w0[0] = 1

            # Run optimization
            res = minimize(sharpe, w0,
                           method='SLSQP',
                           constraints=constraints,
                           bounds=bounds,
                           options={'ftol': 1e-9, 'disp': False})

            if not res.success:
                raise RuntimeError("Convergence Failed")

            # Compute optimal portfolio parameters
            mu_p = np.sum(res.x * self.mu)
            sigma_p = np.sqrt(res.x @ self.cov @ res.x)
            sharpe_p = -sharpe(res.x)
            weights = pd.Series(index=self.mu.index,
                                data=res.x,
                                name='Risky Weights')

        return mu_p, sigma_p, weights, sharpe_p

    def _get_minimal_variance_portfolio(self):

        def risk(x):
            return np.sqrt(x @ self.cov @ x)

        # budget constraint
        constraints = ({'type': 'eq',
                        'fun': lambda w: w.sum() - 1})

        # Create bounds for the weights if short-selling is restricted
        if self.short_selling:
            bounds = None
        else:
            bounds = Bounds(np.zeros(self.n_assets), np.ones(self.n_assets))

        # initial guess
        w0 = np.zeros(self.n_assets)
        w0[0] = 1

        # Run optimization
        res = minimize(risk, w0,
                       method='SLSQP',
                       constraints=constraints,
                       bounds=bounds,
                       options={'ftol': 1e-9, 'disp': False})

        if not res.success:
            raise RuntimeError("Convergence Failed")

        # Compute optimal portfolio parameters
        mu_mv = np.sum(res.x * self.mu)
        sigma_mv = np.sqrt(res.x @ self.cov @ res.x)
        sharpe_mv = (mu_mv - self.rf) / sigma_mv
        weights = pd.Series(index=self.mu.index,
                            data=res.x,
                            name='Minimal Variance Weights')

        return mu_mv, sigma_mv, weights, sharpe_mv

    def _get_cov_matrix(self):

        cov = np.diag(self.sigma) @ self.corr.values @ np.diag(self.sigma)

        cov = pd.DataFrame(index=self.sigma.index,
                           columns=self.sigma.index,
                           data=cov)

        return cov

    def _investor_allocation(self):
        weight_p = (self.mu_p - self.rf) / (self.risk_aversion * (self.sigma_p**2))
        complete_weights = self.risky_weights * weight_p
        complete_weights.loc['Risk Free'] = 1 - weight_p

        mu_c = weight_p * self.mu_p + (1 - weight_p) * self.rf
        sigma_c = weight_p * self.sigma_p

        ce = self._utility(mu_c, sigma_c, self.risk_aversion)

        return weight_p, complete_weights, mu_c, sigma_c, ce

    def _min_var_frontier(self, n_steps=100):
        E = self.mu.values
        inv_cov = inv(self.cov)

        A = E @ inv_cov @ E
        B = np.ones(self.n_assets) @ inv_cov @ E
        C = np.ones(self.n_assets) @ inv_cov @ np.ones(self.n_assets)

        def min_risk(mu):
            return np.sqrt((C * (mu ** 2) - 2 * B * mu + A)/(A * C - B ** 2))

        min_mu = min(self.mu.min(), self.rf) - 0.01  # TODO how to systematize this?
        max_mu = max(self.mu.max(), self.rf) + 0.05  # TODO how to systematize this?

        mu_range = np.arange(min_mu, max_mu, (max_mu - min_mu) / n_steps)
        sigma_range = np.array(list(map(min_risk, mu_range)))

        return mu_range, sigma_range

    @staticmethod
    def _sharpe(w, mu, cov, rf, n):
        er = np.sum(w*mu)

        w = np.reshape(w, (n, 1))
        risk = np.sqrt(w.T @ cov @ w)[0][0]

        sharpe = (er - rf) / risk
        return sharpe

    @staticmethod
    def _utility(mu, sigma, risk_aversion):
        return mu - 0.5 * risk_aversion * (sigma ** 2)

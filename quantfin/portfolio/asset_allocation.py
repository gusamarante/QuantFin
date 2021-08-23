from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from numpy.linalg import inv
from tqdm import tqdm
import pandas as pd
import numpy as np


class Markowitz(object):
    # TODO Make the 'risk_aversion' parameter optional
    # TODO change inputs to covariance

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

        # Asseert data indexes and organize
        self._assert_indexes(mu, sigma, corr)

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

    def plot(self, figsize=None, save_path=None):

        plt.figure(figsize=figsize)
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
            mu_mv, sigma_mv = self.min_var_frontier()
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

        # Save as picture
        if save_path is not None:
            plt.savefig(save_path)

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
            mu_p = np.sum(res.x * self.mu.values.flatten())
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
        mu_mv = np.sum(res.x * self.mu.values.flatten())
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

    def min_var_frontier(self, n_steps=100):

        if self.short_selling:
            E = self.mu.values
            inv_cov = inv(self.cov)

            A = E @ inv_cov @ E
            B = np.ones(self.n_assets) @ inv_cov @ E
            C = np.ones(self.n_assets) @ inv_cov @ np.ones(self.n_assets)

            def min_risk(mu):
                return np.sqrt((C * (mu ** 2) - 2 * B * mu + A)/(A * C - B ** 2))

            min_mu = min(self.mu.min(), self.rf) - 0.01
            max_mu = max(self.mu.max(), self.rf) + 0.05

            mu_range = np.arange(min_mu, max_mu, (max_mu - min_mu) / n_steps)
            sigma_range = np.array(list(map(min_risk, mu_range)))

        else:

            sigma_range = []

            # Objective function
            def risk(x):
                return np.sqrt(x @ self.cov @ x)

            # initial guess
            w0 = np.zeros(self.n_assets)
            w0[0] = 1

            # Values for mu to perform the minimization
            mu_range = np.linspace(self.mu.min(), self.mu.max(), n_steps)

            for mu_step in tqdm(mu_range, 'Finding Mininmal variance frontier'):

                # budget and return constraints
                constraints = ({'type': 'eq',
                                'fun': lambda w: w.sum() - 1},
                               {'type': 'eq',
                                'fun': lambda w: sum(w * self.mu) - mu_step})

                bounds = Bounds(np.zeros(self.n_assets), np.ones(self.n_assets))

                # Run optimization
                res = minimize(risk, w0,
                               method='SLSQP',
                               constraints=constraints,
                               bounds=bounds,
                               options={'ftol': 1e-9, 'disp': False})

                if not res.success:
                    raise RuntimeError("Convergence Failed")

                # Compute optimal portfolio parameters
                sigma_step = np.sqrt(res.x @ self.cov @ res.x)

                sigma_range.append(sigma_step)

            sigma_range = np.array(sigma_range)

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

    @staticmethod
    def _assert_indexes(mu, sigma, corr):
        cond1 = sorted(mu.index) == sorted(sigma.index)
        cond2 = sorted(mu.index) == sorted(corr.index)

        cond = cond1 and cond2

        assert cond, "elements in the input indexes do not match"


class BlackLitterman(object):
    # TODO implement qualitative view-setting

    def __init__(self, sigma, estimation_error, views_p, views_v, w_equilibrium=None, avg_risk_aversion=1.2,
                 mu_historical=None, mu_shrink=1, overall_confidence=1, relative_uncertainty=None):
        """
        Black-Litterman model for asset allocation. The model combines model estimates and views
        from the asset allocators. The views are expressed in the form of

            views_p @ mu = views_v

        where 'views_p' is a selection matrix and 'views_v' is a vector of values.

        :param sigma: pandas.DataFrame, robustly estimated covariance matrix of the assets.
        :param estimation_error: float, Uncertainty of the estimation. Recomended value is
                                 the inverse of the sample size used in the covariance matrix.
        :param views_p: pandas.DataFrame, selection matrix of the views.
        :param views_v: pandas.DataFrame, value matrix of the views.  # TODO allow for pandas.Series
        :param w_equilibrium: pandas.DataFrame, weights of each asset in the equilibrium
        :param avg_risk_aversion: float, average risk aversion of the investors
        :param mu_historical: pandas.DataFrame, historical returns of the asset class (can
                              be interpreted as the target of the shrinkage estimate)
        :param mu_shrink: float between 0 and 1, shirinkage intensity. If 1 (default),
                          best guess of mu is the model returns. If 0, bet guess of mu
                          is 'mu_historical'.  # TODO assert domain of 0 to 1
        :param overall_confidence: float, the higher the number, the more weight the views have in te posterior
        :param relative_uncertainty: pandas.DataFrame, the higher the value the less certain that view is.  # TODO allow for pandas series
        """

        # TODO assert input types (DataFrames)
        # TODO assert input shapes and names
        # TODO assert covariances are positive definite

        self.sigma = sigma.sort_index(0).sort_index(1)
        self.asset_names = list(self.sigma.index)
        self.n_assets = sigma.shape[0]
        self.estimation_error = estimation_error
        self.avg_risk_aversion = avg_risk_aversion
        self.mu_shrink = mu_shrink

        self.w_equilibrium = self._get_w_equilibrium(w_equilibrium)
        self.equilibrium_returns = self._get_equilibrium_returns()
        self.mu_historical = self._get_mu_historical(mu_historical)
        self.mu_best_guess = self._get_mu_best_guess()

        self.views_p = views_p.sort_index(0).sort_index(1)
        self.views_v = views_v.sort_index(0).sort_index(1)
        self.n_views = views_p.shape[0]
        self.view_names = list(self.views_p.index)
        self.overall_confidence = overall_confidence
        self.relative_uncertainty = self._get_relative_uncertainty(relative_uncertainty)
        self.omega = self._get_views_covariance()

        self.mu_bl, self.sigma_mu_bl = self._get_mu_bl()
        self.sigma_bl = self.sigma + self.sigma_mu_bl

    def _get_w_equilibrium(self, w_equilibrium):
        """
        In case 'w_equilibrium' is not passed, assumes the equilibrium is equal weighted.
        """
        if w_equilibrium is None:
            w_equilibrium = (1 / self.n_assets) * np.ones(self.n_assets)
            w_equilibrium = pd.DataFrame(data=w_equilibrium,
                                         index=self.asset_names,
                                         columns=['Equilibrium Weights'])
        else:
            w_equilibrium = w_equilibrium.sort_index()

        return w_equilibrium

    def _get_equilibrium_returns(self):
        """
        Computes the equilibrium returns based on the equilibrium weights and
        average risk aversion.
        """
        sigma = self.sigma.values
        w_equilibrium = self.w_equilibrium.values
        pi = 2 * self.avg_risk_aversion * sigma @ w_equilibrium
        pi = pd.DataFrame(data=pi,
                          index=self.asset_names,
                          columns=['Equilibrium Returns'])
        return pi

    def _get_mu_historical(self, mu_historical):
        """
        In case 'mu_historical' is not passed, uses zeros as the shrinking target.
        """
        if mu_historical is None:
            mu_historical = np.zeros(self.n_assets)
            mu_historical = pd.DataFrame(data=mu_historical,
                                         index=self.asset_names,
                                         columns=['Historical Returns'])
        else:
            mu_historical = mu_historical.sort_index()

        return mu_historical

    def _get_mu_best_guess(self):
        """
        Uses shrinkage to estimate the best guess for mu by balancing between
        the model equilibrium returns and the historical returns.
        """
        best_guess = self.mu_shrink * self.equilibrium_returns.values + (1-self.mu_shrink) * self.mu_historical.values
        best_guess = pd.DataFrame(data=best_guess,
                                  index=self.asset_names,
                                  columns=['Best Guess of mu'])
        return best_guess

    def _get_relative_uncertainty(self, relative_uncertainty):
        """
        In case 'relative_uncertainty' is not passed, uses ones for every asset.
        """
        if relative_uncertainty is None:
            relative_uncertainty = np.ones(self.n_views)
            relative_uncertainty = pd.DataFrame(data=relative_uncertainty,
                                                columns=['Relative Uncertainty'],
                                                index=self.view_names)
        else:
            relative_uncertainty = relative_uncertainty.sort_index()

        return relative_uncertainty

    def _get_views_covariance(self):
        """
        Computes Omega, the covariance of the views.
        """
        c = self.overall_confidence
        u = np.diag(self.relative_uncertainty.values.flatten())
        P = self.views_p.values
        Sigma = self.sigma.values

        omega = (1/c) * u @ P @ Sigma @ P.T @ u

        if np.linalg.det(omega) < np.finfo(float).eps:
            n, m = omega.shape
            omega = omega + 1e-16 * np.eye(n, m)

        omega = pd.DataFrame(data=omega,
                             index=self.view_names,
                             columns=self.view_names)

        return omega

    def _get_mu_bl(self):
        """
        Computes 'mu_bl', the vector of returns that combines the best guess
        for mu (equilibrium and empirical) with the views from the asset allocators
        """
        tau = self.estimation_error
        sigma = self.sigma.values
        P = self.views_p.values
        pi = self.mu_best_guess.values
        v = self.views_v.values
        omega = self.omega.values

        sigma_mu_bl = inv(inv(tau * sigma) + P.T @ inv(omega) @ P)
        B = inv(tau * sigma) @ pi + P.T @ inv(omega) @ v
        mu_bl = sigma_mu_bl @ B

        sigma_mu_bl = pd.DataFrame(data=sigma_mu_bl, index=self.asset_names, columns=self.asset_names)
        mu_bl = pd.DataFrame(data=mu_bl, index=self.asset_names, columns=['Expected Returns'])

        return mu_bl, sigma_mu_bl

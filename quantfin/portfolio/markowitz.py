import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds


class Markowitz(object):
    # TODO Minumum Variance Frontier
    # TODO Chart Functionality
    # TODO Borrowing Rate
    # TODO Documentation

    def __init__(self, mu, sigma, corr, rf, risk_aversion=None, short_sell=True):
        # TODO Assert data types
        # TODO Assert data indexes match

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
                return -self._sharpe(x, self.mu.values, self.cov.values, self.rf)

            # budget constraint
            constraints = [{'type': 'eq',
                           'fun': lambda w: w.sum() - 1}]

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
        constraints = [{'type': 'eq',
                        'fun': lambda w: w.sum() - 1}]

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

    @staticmethod
    def _sharpe(w, mu, cov, rf):
        er = np.sum(w*mu)

        w = np.reshape(w, (2, 1))
        risk = np.sqrt(w.T @ cov @ w)[0][0]

        sharpe = (er - rf) / risk
        return sharpe

    @staticmethod
    def _utility(mu, sigma, risk_aversion):
        return mu - 0.5 * risk_aversion * (sigma ** 2)

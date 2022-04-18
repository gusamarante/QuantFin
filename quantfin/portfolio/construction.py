"""
Classes for porfolio construction
"""
import numpy as np
import pandas as pd
from numpy.linalg import inv, eig


class EqualWeights(object):

    def __init__(self, df):
        self.weights = (~df.isna()).div(df.count(axis=1), axis=0)
        self.returns = (self.weights * df.pct_change(1).dropna(how='all')).sum(axis=1)
        self.return_index = 100 * (1 + self.returns).cumprod()
        self.return_index.name = 'Equal Weighted'


class PrincipalPortfolios(object):
    """
    Implementation of the 'Principal Portfolios'.
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3623983
    """

    def __init__(self, returns, signals):
        # TODO covariance shirinkage using Eigenvalue reconstruction
        """
        [DESCRIPTION HERE OF ALL THE ATTRIBUTES]
        :param returns:
        :param signals: Should already have the appropriate lag.
        """

        self.asset_names = list(returns.columns)
        self.asset_number = len(self.asset_names)
        self.returns, self.signals = self._trim_dataframes(returns, signals)
        self.pred_matrix = self._get_prediction_matrix()
        self.cov_returns = self._get_covariance_returns()

        # Principal Portfolios (PP)
        self.svd_left, self.svd_values, self.svd_right = self._get_svd()
        self.er_pp = self.svd_values.sum()  # equivalent to tr(L @ PI)
        self.optimal_selection = self.svd_right @ self.svd_left.T  # paper calls this L, proposition 3, pg 13
        self.optimal_weights = self._get_optimal_weights()

        # Latent factor
        self.factor_weights = self._get_factor_weights()

        # Symmetry decompositions
        self.pi_s, self.pi_a = self._get_symmetry_separation(self.pred_matrix)

        # Principal Exposure Portfolios (PEP) - Symmetric Strategies
        self.pi_s_eigval, self.pi_s_eigvec = self._get_symmetric_eig()

        # Principal Alpha Portfolios (PAP) - Anti-symmetric Strategies
        self.pi_a_eigval, self.pi_a_eigvec = self._get_antisymmetric_eig()

    def get_pp(self, k=1):
        """
        Gets the weights of k-th principal portfolio, shown in euqation 15 of the paper.
        :param k: int. The number of the desired principal portfolio.
        :return: tuple. First entry are the weights, second is the selection matrix and third is the singular
                 value, which can be interpreted as the expected return (proposition 4).
        """

        assert k <= self.asset_number, "'k' must not be bigger than then number of assets"

        uk = self.svd_left[:, k - 1].reshape((-1, 1))
        vk = self.svd_right[:, k - 1].reshape((-1, 1))
        s = self.signals.iloc[-1].values
        singval = self.svd_values[k - 1]

        lk = vk @ uk.T
        wk = (s.T @ lk)
        wk = pd.Series(index=self.asset_names, data=wk, name=f'PP {k}')

        return wk, lk, singval

    def get_pep(self, k=1, absolute=True):
        """
        Gets the weights of k-th principal exposure portfolio (PEP), shown in equation 30 of the paper.
        :param k: int. The number of the desired principal exposure portfolio.
        :param absolute: If eigenvalues should be sorted on absolute value or not. Default is true, to get the
                         PEPs in order of expected return.
        :return: tuple. First entry are the weights, second is the selection matrix and third is the eigenvalue,
                 which can be interpreted as the expected return (proposition 6).
        """
        assert k <= self.asset_number, "'k' must not be bigger than then number of assets"

        eigval, eigvec = self.pi_s_eigval, self.pi_s_eigvec
        s = self.signals.iloc[-1].values

        if absolute:
            signal = np.sign(eigval)
            eigvec = eigvec * signal  # Switch the signals of the eigenvectors with negative eigenvalues
            eigval = np.abs(eigval)
            idx = eigval.argsort()[::-1]  # re sort eigenvalues based on absolute value and the associated eigenvectors
            eigval = eigval[idx]
            eigvec = eigvec[:, idx]

        vsk = eigvec[:, k - 1].reshape((-1, 1))  # from equation 30
        lsk = vsk @ vsk.T
        wsk = s.T @ lsk
        wsk = pd.Series(data=wsk, index=self.asset_names, name=f'PEP {k}')
        return wsk, lsk, eigval[k - 1]

    def get_pap(self, k=1):
        """
        Gets the weights of k-th principal alpha portfolio (PAP), shown in equation 35 of the paper.
        :param k: int. The number of the desired principal alpha portfolio.
        :return: tuple. First entry are the weights, second is the selection matrix and third is the
                 eigenvalue times 2, which can be interpreted as the expected return (proposition 8).
        """
        assert k <= self.asset_number/2, "'k' must not be bigger than then half of the number of assets"

        eigval, eigvec = self.pi_a_eigval, self.pi_a_eigvec
        s = self.signals.iloc[-1].values

        v = eigvec[:, k - 1].reshape((-1, 1))
        x = v.real
        y = v.imag
        l = x @ y.T - y @ x.T
        w = s.T @ l
        w = pd.Series(data=w, index=self.asset_names, name=f'PAP {k}')
        return w, l, 2 * eigval[k - 1]

    def _get_prediction_matrix(self):
        size = self.returns.shape[0]
        # dev_mat = np.eye(size) - np.ones((size, size)) * (1 / size)
        pi = (1 / size) * (self.returns.values.T @ self.signals.values)
        return pi

    def _get_optimal_weights(self):
        s = self.signals.iloc[-1].values
        l = self.optimal_selection
        w = s.dot(l)  # paper calls this S'L
        w = pd.Series(index=self.asset_names, data=w)
        return w

    def _get_svd(self):
        pi = self.pred_matrix
        u, sing_values, vt = np.linalg.svd(pi)
        return u, sing_values, vt.T

    def _get_covariance_returns(self):
        cov = self.returns.cov()
        return cov

    def _get_factor_weights(self):
        cov = self.cov_returns.values
        s = self.signals.iloc[-1].values
        factor_weights = ((s @ inv(cov) @ s)**(-1)) * (inv(cov) @ s)
        factor_weights = pd.Series(data=factor_weights, index=self.asset_names, name='Factor Weights')
        return factor_weights

    def _get_symmetric_eig(self):
        eigval, eigvec = eig(self.pi_s)
        idx = eigval.argsort()[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]
        return eigval, eigvec

    def _get_antisymmetric_eig(self):
        eigval, eigvec = eig(self.pi_a.T)
        eigval = eigval.imag  # Grabs the imaginary part. The real part is zero, but with numerical error.
        idx = eigval.argsort()[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]
        return eigval, eigvec

    @staticmethod
    def _get_symmetry_separation(mat):
        mat_s = 0.5 * (mat + mat.T)
        mat_a = 0.5 * (mat - mat.T)
        return mat_s, mat_a

    @staticmethod
    def _trim_dataframes(returns, signals):
        start_returns = returns.index[0]
        start_signals = signals.index[0]

        if start_returns >= start_signals:
            signals = signals.reindex(returns.index)
        else:
            returns = returns.reindex(signals.index)

        return returns, signals

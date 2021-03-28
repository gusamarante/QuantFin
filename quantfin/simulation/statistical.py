import numpy as np


def random_covariance(n_col, n_factors, random_seed=None):
    # TODO Documentation
    # TODO Notebook example
    if random_seed is not None:
        np.random.seed(random_seed)

    w = np.random.normal(size=(n_col, n_factors))
    cov = w @ w.T  # random covariance but not with full rank
    cov = cov + np.diag(np.random.uniform(size=n_col))  # make it full rank
    return cov


def random_correlation(n_col, n_factors, random_seed=None):
    # TODO Documentation
    # TODO Notebook example
    cov = random_covariance(n_col, n_factors, random_seed)
    corr = cov2corr(cov)
    return corr


def cov2corr(cov):
    # TODO Documentation
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1] = -1  # correct for numerical error
    corr[corr > 1] = 1
    return corr

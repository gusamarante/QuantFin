import numpy as np
from quantfin.statistics import cov2corr


def random_covariance(n_col, n_factors, random_seed=None):
    # TODO Documentation
    if random_seed is not None:
        np.random.seed(random_seed)

    w = np.random.normal(size=(n_col, n_factors))
    cov = w @ w.T  # random covariance but not with full rank
    cov = cov + np.diag(np.random.uniform(size=n_col))  # make it full rank
    return cov


def random_correlation(n_col, n_factors, random_seed=None):
    # TODO Documentation
    cov = random_covariance(n_col, n_factors, random_seed)
    corr = cov2corr(cov)
    return corr

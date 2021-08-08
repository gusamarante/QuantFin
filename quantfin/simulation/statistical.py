import numpy as np
from quantfin.statistics import cov2corr


def random_covariance(size, n_factors, random_seed=None):
    """
    Generates a random covariance matrix with 'size' lines and columns and
    'n_factors' factors in the underlying structure of covariance.
    :param size: int. Size of the covariance matrix
    :param n_factors: int. number of factors in the covariance structure
    :param random_seed: int. random seed number
    :return: numpy.array. correlation matrix
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    w = np.random.normal(size=(size, n_factors))
    cov = w @ w.T  # random covariance but not with full rank
    cov = cov + np.diag(np.random.uniform(size=size))  # make it full rank
    return cov


def random_correlation(size, n_factors, random_seed=None):
    """
    Generates a random correlation matrix with 'size' lines and columns and
    'n_factors' factors in the underlying structure of correlation.
    :param size: int. Size of the correlation matrix
    :param n_factors: int. number of factors in the correlation structure
    :param random_seed: int. random seed number
    :return: numpy.array. correlation matrix
    """
    cov = random_covariance(size, n_factors, random_seed)
    corr = cov2corr(cov)
    return corr

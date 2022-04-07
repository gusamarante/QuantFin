import numpy as np


def cov2corr(cov):
    """
    Given a covariance matrix, it returns its correlation matrix.
    :param cov: numpy.array covariance matrix
    :return: numpy.array correlation matrix
    """

    assert np.all(np.linalg.eigvals(cov) > 0), "'cov' matrix is not positive definite"
    assert cov.shape[0] == cov.shape[1], "'cov' matrix is not square"

    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1] = -1  # correct for numerical error
    corr[corr > 1] = 1
    return corr, std


def corr2cov(corr, std):
    """
    Given a correlation matrix and a vector of standard deviations, it returns the covariance matrix.
    :param corr: numpy.array correlation matrix
    :param std: numpy.array vector of standard deviations
    :return: numpy.array covariance matrix
    """

    assert np.all(np.linalg.eigvals(corr) > 0), "'cov' matrix is not positive definite"
    assert np.all(std >= 0), "'std' must not contain negative numbers"
    assert corr.shape[0] == corr.shape[1], "'cov' matrix is not square"

    cov = np.diag(std) @ corr @ np.diag(std)
    return cov

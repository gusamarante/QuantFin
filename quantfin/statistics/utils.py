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


def empirical_correlation(df):
    """
    This functions just excludes the observations that are not available before computing the correlation matrix.
    This makes sures that the resulting matrix is positive definite.
    :param df: pandas.DataFrame with the series that are going to be in the correlation matrix.
    :return: pandas.DataFrame with the correlation matrix.
    """
    corr = df.dropna().corr()
    return corr


def rescale_vol(df, target_vol=0.1):
    """
    Rescale return indexes (total or excess) to have the desired volatitlity.
    :param df: pandas.DataFrame of return indexes.
    :param target_vol: float with the desired volatility.
    :return: pandas.DataFrame with rescaled total return indexes.
    """
    returns = df.pct_change(1, fill_method=None)
    returns_std = returns.std() * np.sqrt(252)
    returns = target_vol * returns / returns_std

    df_adj = (1 + returns).cumprod()
    df_adj = 100 * df_adj / df_adj.fillna(method='bfill').iloc[0]

    return df_adj

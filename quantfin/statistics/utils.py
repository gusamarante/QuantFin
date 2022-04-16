import numpy as np
import pandas as pd


def is_psd(mat):
    ans = np.all(np.linalg.eigvals(mat) >= 0)
    return ans


def make_psd(mat, method='abseig'):
    """
    Differnet regularization methods for positive semi-definite matrices (PSD). They find the "closest" PSD
    matrix based on chosen method. This is particularly useful when large estimated covariance matrices
    end up with numerical errors that turn them into non-PSD. Notice that this should not be used on
    correlation matrices, as the main diagonal has to remain equal to 1. The implemented methods are:

    - 'abseig': Uses the eigenvalue-eigenvector decomposition and uses the absolute value of the
                eigenvalues to reconstruct the original matrix.
    - 'frobenius': nearest SPD matrix based on the Frobenius norm.
                   https://www.sciencedirect.com/science/article/pii/0024379588902236?via%3Dihub

    :param mat: numpy.array, matrix to be turned into PSD
    :param method: str, name of the regularization method
    """
    # TODO make sure inputs as DataFrame come out as DataFrame

    assert method in ['abseig', 'frobenius'], 'method not implemented'
    assert mat.shape[0] == mat.shape[1], "'mat' must be a square matrix"

    if method == 'abseig':
        val, vec = np.linalg.eig(mat)
        new_mat = vec @ np.diag(np.abs(val)) @ vec.T

    elif method == 'frobenius':
        B = (mat + mat.T) / 2
        _, s, V = np.linalg.svd(B)
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        if is_psd(A3):
            new_mat = A3
        else:
            spacing = np.spacing(np.linalg.norm(mat))
            I = np.eye(mat.shape[0])
            k = 1
            while not is_psd(A3):
                mineig = np.min(np.real(np.linalg.eigvals(A3)))
                A3 += I * (-mineig * k ** 2 + spacing)
                k += 1

            new_mat = A3

    else:
        raise NotImplementedError('method not implemented')

    return new_mat


def cov2corr(cov):
    """
    Given a covariance matrix, it returns its correlation matrix.
    :param cov: numpy.array covariance matrix
    :return: numpy.array correlation matrix
    """

    assert np.all(np.linalg.eigvals(cov) >= 0), "'cov' matrix is not positive semi-definite"
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

    corr = np.array(corr)
    std = np.array(std)

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

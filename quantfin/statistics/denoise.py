import numpy as np
import pandas as pd
from scipy.optimize import minimize
from quantfin.statistics import cov2corr
from sklearn.neighbors import KernelDensity


# ===== Marchenko-Pastur Denoising =====
def marchenko_pastur(corr_matrix, T, N, bandwidth=0.1):
    # TODO allow input to be covariance (can I check if diagonal is all ones?)
    """
    Uses the Marchenko-Pastur theorem to remove noisy eigenvalues from a correlation matrix.
    This code is adapted from Lopez de Prado (2020).
    :param corr_matrix: numpy.array. Correlation matrix from data.
    :param T: int. Sample size of the timeseries dimensions.
    :param N: int. Sample size of the cross-section dimensions.
    :param bandwidth: smoothing parameter for the KernelDensity estimation
    :return: 'corr' is the denoised correlation matrix, 'nFacts' is the number of non-random
             factors in the original correlation matrix and 'var' is the estimate of sigma**2,
             which can be interpreted as the % of noise in the original correlationm matrix.
    """

    # get eigenvalues and eigenvectors
    eVal, eVec = np.linalg.eigh(corr_matrix)
    indices = eVal.argsort()[::-1]
    eVal, eVec = eVal[indices], eVec[:, indices]
    eVal = np.diagflat(eVal)

    # find sigma that minimizes the error to the Marchenko-Pastur distribution
    q = T / N
    eMax, var = _find_max_eigval(np.diag(eVal), q, bWidth=bandwidth)

    # number of factors (signals)
    nFacts = eVal.shape[0] - np.diag(eVal)[::-1].searchsorted(eMax)

    eVal_ = np.diag(eVal).copy()
    eVal_[nFacts:] = eVal_[nFacts:].sum() / float(eVal_.shape[0] - nFacts)
    eVal_ = np.diag(eVal_)
    cov = np.dot(eVec, eVal_).dot(eVec.T)
    corr = cov2corr(cov)

    return corr, nFacts, var


def targeted_shirinkage(corr_matrix, T, N, bandwidth=0.1, ts_alpha=None):
    """
    Uses the Marchenko-Pastur theorem to find noisy eigenvalues from a correlation matrix and
    performs shrinkage only on the noisy part of the correlation matrix. This code is adapted
    from Lopez de Prado (2020).
    :param corr_matrix: numpy.array. Correlation matrix from data.
    :param T: int. Sample size of the timeseries dimensions.
    :param N: int. Sample size of the cross-section dimensions.
    :param bandwidth: smoothing parameter for the KernelDensity estimation
    :param ts_alpha: float. Number between 0 and 1 indicating the ammount of targeted shrinkage
                     on the random eigenvectors. ts_alpha=0 means no shrinkage and ts_alpha=1
                     means total shrinkage.
    :return: 'corr' is the denoised correlation matrix, 'nFacts' is the number of non-random
             factors in the original correlation matrix and 'var' is the estimate of sigma**2,
             which can be interpreted as the % of noise in the original correlationm matrix.
    """

    # assertions
    if ts_alpha is not None:
        assert 0 <= ts_alpha <= 1, "targeted shrinkage parameter must be between zero and 1."

    # get eigenvalues and eigenvectors
    eVal, eVec = np.linalg.eigh(corr_matrix)
    indices = eVal.argsort()[::-1]
    eVal, eVec = eVal[indices], eVec[:, indices]
    eVal = np.diagflat(eVal)

    # find sigma that minimizes the error to the Marchenko-Pastur distribution
    q = T / N
    eMax, var = _find_max_eigval(np.diag(eVal), q, bWidth=bandwidth)

    # number of factors (signals)
    nFacts = eVal.shape[0] - np.diag(eVal)[::-1].searchsorted(eMax)

    # targeted shrinkage
    eValL, eVecL = eVal[:nFacts, :nFacts], eVec[:, :nFacts]
    eValR, eVecR = eVal[nFacts:, nFacts:], eVec[:, nFacts:]
    corrL = np.dot(eVecL, eValL).dot(eVecL.T)
    corrR = np.dot(eVecR, eValR).dot(eVecR.T)
    corr = corrL + (1 - ts_alpha) * corrR + ts_alpha * np.diag(np.diag(corrR))

    return corr, nFacts, var


def _marchenko_pastur_pdf(var, q, pts):
    eMin = var * (1 - (1. / q) ** .5) ** 2
    eMax = var * (1 + (1. / q) ** .5) ** 2
    eVal = np.linspace(eMin, eMax, pts)
    pdf = q / (2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin)) ** .5
    pdf = pd.Series(pdf.flatten(), index=eVal.flatten())
    return pdf


def _fit_kde(observations, bandwidth, x=None):

    if len(observations.shape) == 1:
        observations = observations.reshape(-1, 1)

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(observations)

    if x is None:
        x = np.unique(observations).reshape(-1, 1)

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    logProb = kde.score_samples(x)  # log(density)
    pdf = pd.Series(np.exp(logProb), index=x.flatten())

    return pdf


def _error_pdfs(var, eVal, q, bandwidth):
    pts = 10 * eVal.shape[0]
    pdf0 = _marchenko_pastur_pdf(var, q, pts)  # theoretical pdf
    pdf1 = _fit_kde(eVal, bandwidth, x=pdf0.index.values)  # empirical pdf
    sse = np.sum((pdf1 - pdf0) ** 2)
    return sse


def _find_max_eigval(eVal, q, bWidth):
    # Finds the maximum random eigenvalue by fitting Marcenko-Pastur distribution
    x0 = np.array([0.5])
    out = minimize(lambda *x: _error_pdfs(*x), x0, args=(eVal, q, bWidth), bounds=[(1E-5, 1 - 1E-5)])

    if out.success:
        var = out.x[0]
    else:
        var = 1

    eMax = var * (1 + (1. / q) ** .5) ** 2
    return eMax, var


# ===== detoning correlation matrix =====
def detone(corr, n=1):
    """
    Removes the first `n` components of the correlation matrix. The detoned correlation matrix
    is singular. This is not a problem for clustering applications as most approaches do not
    require invertibility.
    :param corr: numpy array. Correlation matrix.
    :param n: int. number of the first 'n' components to be removed from the correlation matrix.
    :return: numpy array
    """
    # TODO Notebook example
    # TODO Allow covariance input
    eVal, eVec = np.linalg.eigh(corr)
    indices = eVal.argsort()[::-1]
    eVal, eVec = eVal[indices], eVec[:, indices]
    eVal = np.diagflat(eVal)

    # eliminate the first n eigenvectors
    eVal = eVal[n:, n:]
    eVec = eVec[:, n:]
    corr_aux = np.dot(eVec, eVal).dot(eVec.T)
    corr_d = corr_aux @ np.linalg.inv(np.diag(np.diag(corr_aux)))

    if isinstance(corr, pd.DataFrame):
        corr_d = pd.DataFrame(data=corr_d, index=corr.index, columns=corr.columns)

    return corr_d


# ===== Shrinking the Covariance Matrix =====
def shrink_cov(cov, alpha=0.1):
    """
    Applies shirinkage to the covariance matrix without changing the variance of each factor. This
    method differs from sklearn's method as this preserves the main diagonal of the covariance matrix,
    making this a more suitable method for financial data.
    :param cov: numpy array. Empirical Covariance matrix
    :param alpha: float. A number between 0 and 1 that represents the shrinkage intensity.
    :return: numpy array. Shrunk Covariance matrix.
    """
    # TODO Example

    assert 0 <= alpha <= 1, "'alpha' must be between 0 and 1"

    vols = np.sqrt(np.diag(cov))
    corr, _ = cov2corr(cov)
    shrunk_corr = (1 - alpha) * corr + alpha * np.eye(corr.shape[0])
    shrunk_cov = np.diag(vols) @ shrunk_corr @ np.diag(vols)

    if isinstance(cov, pd.DataFrame):
        shrunk_cov = pd.DataFrame(data=shrunk_cov.values, index=cov.index, columns=cov.columns)

    return shrunk_cov

# ===== Ledoit-Wolfe =====


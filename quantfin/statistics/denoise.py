import numpy as np
import pandas as pd
from scipy.optimize import minimize
from quantfin.statistics import cov2corr
from sklearn.neighbors import KernelDensity


# ===== Marchenko-Pastur Denoising =====
def denoise_corr_mp(corr_matrix, T, N, bandwidth=0.1, ts_alpha=None):
    """
    Uses the Marchenko-Pastur theorem to remove noisy eigenvalues from a correlation matrix.
    This code is adapted from Lopez de Prado (2020)
    :param corr_matrix: numpy.array. Correlation matrix from data.
    :param T: int. Sample size of the timeseries dimensions.
    :param N: int. Sample size of the cross-section dimensions.
    :param bandwidth: smoothing parameter for the KernelDensity estimation
    :param ts_alpha: float. Number between 0 and 1 indicating the ammount of targeted shrinkage
                     on the random eigenvectors. ts_alpha=1 means no shrinkage and ts_alpha=0
                     means total shrinkage.
    :return: 'corr' is the denoised correlation matrix, 'nFacts' is the number of non-random
             factors in the original correlation matrix and 'var' is the estimate of sigma**2,
             which can be interpreted as the % of noise in the original correlationm matrix.
    """

    # assertions
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

    # de-noise the correlation matrix
    if ts_alpha is None:
        eVal_ = np.diag(eVal).copy()
        eVal_[nFacts:] = eVal_[nFacts:].sum() / float(eVal_.shape[0] - nFacts)
        eVal_ = np.diag(eVal_)
        cov = np.dot(eVec, eVal_).dot(eVec.T)
        corr = cov2corr(cov)
    else:
        # targeted shrinkage
        eValL, eVecL = eVal[:nFacts][:nFacts], eVec[:, :nFacts]
        eValR, eVecR = eVal[nFacts:][nFacts:], eVec[:, nFacts:]
        corrL = np.dot(eVecL, eValL).dot(eVecL.T)
        corrR = np.dot(eVecR, eValR).dot(eVecR.T)
        corr = corrL + ts_alpha * corrR + (1-ts_alpha)*np.diag(np.diag(corrR))

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
    # TODO Documentaion
    # TODO Notebook example
    eVal, eVec = np.linalg.eigh(corr)
    indices = eVal.argsort()[::-1]
    eVal, eVec = eVal[indices], eVec[:, indices]
    eVal = np.diagflat(eVal)

    # eliminate the first n eigenvectors
    eVal = eVal[n:, n:]
    eVec = eVec[:, n:]
    corr_aux = np.dot(eVec, eVal).dot(eVec.T)
    corr_d = corr_aux @ np.linalg.inv(np.diag(np.diag(corr_aux)))
    return corr_d

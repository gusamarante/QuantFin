import numpy as np


def make_psd(mat, method='abseig'):
    """
    Differnet regularization methods for positive semi-definite matrices (PSD). They find the "closest" PSD
    matrix based on chosen method. This is particularly useful when large estimated covariance matrices
    end up with numerical errors that turn them into non-PSD. Notice that this should not be used on
    correlation matrices, as the main diagonal has to remain equal to 1. The implemented methods are:

    - 'abseig': Uses the eigenvalue-eigenvector decomposition and uses the absolute value of the
                eigenvalues to reconstruct the original matrix.

    :param mat: numpy.array, matrix to be turned into PSD
    :param method: str, name of the regularization method
    """

    assert method in ['abseig'], 'method not implemented'
    assert mat.shape[0] == mat.shape[1], "'mat' must be a square matrix"

    if method == 'abseig':
        val, vec = np.linalg.eig(mat)
        new_mat = vec @ np.diag(np.abs(val)) @ vec.T
    else:
        raise NotImplementedError('method not implemented')

    return new_mat

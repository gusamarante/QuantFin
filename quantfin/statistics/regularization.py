import numpy as np


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

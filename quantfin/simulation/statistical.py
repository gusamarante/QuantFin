from scipy.stats import random_correlation
import numpy as np


def random_correl(eigenvalues, random_seed=None):
    # TODO Documentation

    if random_seed is not None:
        np.random.seed(random_seed)

    eigenvalues = np.array(eigenvalues)

    cond = (eigenvalues >= 0).all()
    assert cond, "All eigenvalues must be positive"

    dim = len(eigenvalues)

    # Eigenvalues must add up to 'dim'
    eigenvalues = dim * eigenvalues / np.sum(eigenvalues)
    corr = random_correlation.rvs(eigenvalues)

    return corr

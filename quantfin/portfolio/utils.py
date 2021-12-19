import numpy as np


def set_volatility(weights, covariance, target_vol):
    new_weights = target_vol/(np.sqrt(weights @ covariance @ weights))
    return new_weights

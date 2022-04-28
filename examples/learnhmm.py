from quantfin.statistics import GaussianHMM
import matplotlib.pyplot as plt
from hmmlearn import hmm
import numpy as np

# ===== Super basic example =====
np.random.seed(42)

model = hmm.GaussianHMM(n_components=3, covariance_type="full")

# 3 states
model.startprob_ = np.array([0.6, 0.3, 0.1])

model.transmat_ = np.array([[0.7, 0.2, 0.1],
                            [0.3, 0.5, 0.2],
                            [0.3, 0.3, 0.4]])

# 2 Variables
model.means_ = np.array([[0.0, 0.0],
                         [3.0, -3.0],
                         [5.0, 10.0]])

model.covars_ = np.tile(np.identity(2), (3, 1, 1))

# Simulate data
X, Z = model.sample(1000)

# Estimate the model
myhmm = GaussianHMM(X)
myhmm.select_order(show_chart=True, select_iter=5, max_state_number=8)

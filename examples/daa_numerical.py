from quantfin.statistics import GaussianHMM
from quantfin.portfolio import DAACosts
from hmmlearn import hmm
import numpy as np

notional = 100

# ===== Simulate Data =====
startprob = np.array([0.9, 0.1])

transmat = np.array([[0.5, 0.5],
                     [0.5, 0.5]])

means = np.array([[0.05, 0.03],
                  [0.03, 0.05]])

covars = np.array([[[0.01, 0.0],
                    [0.0, 0.01]],

                   [[0.01, 0.0],
                    [0.0, 0.01]]])

# Build an HMM instance and set parameters
gen_model = hmm.GaussianHMM(n_components=2, covariance_type="full")

# Instead of fitting it from the data, we directly set the estimated
gen_model.startprob_ = startprob
gen_model.transmat_ = transmat
gen_model.means_ = means
gen_model.covars_ = covars

# Generate samples
X, Z = gen_model.sample(4000)

# ===== Estimate the HMM =====
hmm = GaussianHMM(X)
hmm.fit(n_states=2, fit_iter=100)

# ===== Compute Allocations =====
start_alloc = notional * np.array([1.50, 1.50])
Lambda1 = (0.0000000001 / 10000) * np.eye(X.shape[1])
Lambda2 = (0.0000000001 / 10000) * np.eye(X.shape[1])
Lambda = np.array([Lambda1, Lambda2])

daa = DAACosts(means=hmm.means,
               covars=hmm.covars,
               costs=Lambda,
               transition_matrix=hmm.trans_mat,
               current_allocation=start_alloc,
               risk_aversion=0.0266667431442302,
               discount_factor=0.99,
               include_returns=True,
               normalize=False)

print('Model Allocations')
print(daa.allocations, '\n')

print('Aim Portfolios')
print(daa.aim_portfolios, '\n')

print('Mkw Portfolios')
print(daa.markowitz_portfolios, '\n')


daa.allocations.to_clipboard()

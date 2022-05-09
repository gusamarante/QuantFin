from quantfin.statistics import GaussianHMM
from quantfin.portfolio import DAACosts
from hmmlearn import hmm
import numpy as np

# ===== Simulate Data =====
startprob = np.array([0.6, 0.4])

transmat = np.array([[0.9, 0.1],
                     [0.55, 0.45]])

means = np.array([[0.02, 0.005],
                  [-0.01, 0.04]])

covars = np.array([[[0.0016, 0],
                    [0, 0.000625]],

                   [[0.01, -0.0015],
                    [-0.0015, 0.0025]]])

# Build an HMM instance and set parameters
gen_model = hmm.GaussianHMM(n_components=2, covariance_type="full")

# Instead of fitting it from the data, we directly set the estimated
gen_model.startprob_ = startprob
gen_model.transmat_ = transmat
gen_model.means_ = means
gen_model.covars_ = covars

# Generate samples
X, Z = gen_model.sample(1000)

# ===== Estimate the HMM =====
hmm = GaussianHMM(X)
hmm.fit(n_states=2, fit_iter=100)

# ===== Compute Allocations =====
start_alloc = 40000000 * np.array([0.2, 0.8])
Lambda1 = (5 / 10000) * np.eye(X.shape[1])
Lambda2 = (10 / 10000) * np.eye(X.shape[1])
Lambda = np.array([Lambda1, Lambda2])

daa = DAACosts(means=hmm.means,
               covars=hmm.covars,
               costs=Lambda,
               transition_matrix=hmm.trans_mat,
               current_allocation=start_alloc,
               risk_aversion=1,
               discount_factor=0.99,
               include_returns=True,
               normalize=True)

print('Model Allocations')
print(daa.allocations, '\n')

print('Aim Portfolios')
print(daa.aim_portfolios, '\n')

print('Mkw Portfolios')
print(daa.markowitz_portfolios, '\n')

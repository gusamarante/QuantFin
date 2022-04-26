import matplotlib.pyplot as plt
from hmmlearn import hmm
import numpy as np

# Prepare parameters for a 4-state / 2-variable
# Initial population probability
startprob = np.array([0.6, 0.3, 0.1, 0.0])

# The transition matrix, note that there are no transitions possible between component 1 and 3
transmat = np.array([[0.7, 0.2, 0.0, 0.1],
                     [0.3, 0.5, 0.2, 0.0],
                     [0.0, 0.3, 0.5, 0.2],
                     [0.2, 0.0, 0.2, 0.6]])

# The means of each component
means = np.array([[0.0, 0.0],
                  [0.0, 11.0],
                  [9.0, 10.0],
                  [11.0, -1.0]])

# The covariance of each component
# covars = .5 * np.tile(np.identity(2), (4, 1, 1))

covars = np.array([[[1., 0.9],  # Variables should be positively correlated in state 1
                    [0.9, 1.]],

                   [[1., -0.9],  # and negatively correlated in state 2
                    [-0.9, 1.]],

                   [[1., 0.],  # and not correlated in states 3 and 4
                    [0., 1.]],

                   [[1., 0.],
                    [0., 1.]]])

# Build an HMM instance and set parameters
gen_model = hmm.GaussianHMM(n_components=4, covariance_type="full", verbose=True)

# Instead of fitting it from the data, we directly set the estimated
# parameters, the means and covariance of the components
gen_model.startprob_ = startprob
gen_model.transmat_ = transmat
gen_model.means_ = means
gen_model.covars_ = covars

# Generate samples
X, Z = gen_model.sample(500)

# Plot the sampled data
fig, ax = plt.subplots()
ax.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6,
        mfc="orange", alpha=0.7)

# Indicate the component numbers
for i, m in enumerate(means):
    ax.text(m[0], m[1], 'State %i' % (i + 1),
            size=7, horizontalalignment='center',
            bbox=dict(alpha=.7, facecolor='w'))

plt.show()

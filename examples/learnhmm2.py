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

covars = np.array([[[1., 0.8],  # Variables should be positively correlated in state 1
                    [0.8, 1.]],

                   [[1., -0.8],  # and negatively correlated in state 2
                    [-0.8, 1.]],

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
X, Z = gen_model.sample(5000)

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

# Estimate the parameters
scores = list()
models = list()
plotx = list()
for n_components in range(2, 9):
    # define our hidden Markov model
    model = hmm.GaussianHMM(n_components=n_components,
                            covariance_type='full', n_iter=10)
    model.fit(X[:X.shape[0] // 2])  # 50/50 train/validate
    models.append(model)
    scores.append(model.score(X[X.shape[0] // 2:]))
    plotx.append(n_components)
    print(f'Converged: {model.monitor_.converged}'
          f'\tScore: {scores[-1]}')


# get the best model
model = models[np.argmax(scores)]
n_states = model.n_components
print(f'The best model had a score of {max(scores)} and {n_states} '
      'states')

plt.plot(plotx, scores)
plt.show()

# use the Viterbi algorithm to predict the most likely sequence of states
# given the model
states = model.predict(X)

# plot model states over time
fig, ax = plt.subplots()
ax.plot(Z, states)
ax.set_title('States compared to generated')
ax.set_xlabel('Generated State')
ax.set_ylabel('Recovered State')
fig.show()

# plot the transition matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
ax1.imshow(gen_model.transmat_, aspect='auto', cmap='spring')
ax1.set_title('Generated Transition Matrix')
ax2.imshow(model.transmat_, aspect='auto', cmap='spring')
ax2.set_title('Recovered Transition Matrix')
for ax in (ax1, ax2):
    ax.set_xlabel('State To')
    ax.set_ylabel('State From')

plt.tight_layout()
plt.show()

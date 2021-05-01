import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve

show_charts = False
save_path = '/Users/gustavoamarante/Dropbox/Aulas/QuantFin/figures/'

k_test = 51
h = 0.02  # step size in the mesh

# ===== GENERATE DATA =====
X, y = make_moons(noise=0.3,
                  random_state=444,
                  n_samples=200)

# plot the data
cmap_light = ListedColormap(['#FFBC00', '#5FC4FF'])
cmap_bold = ['#FF570A', '#2F34AF']

plt.figure(figsize=(6, 6))
ax = plt.gca()
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(cmap_bold), edgecolor="black", label=None)
ax.set(xlabel='$X$', ylabel='$Y$')
plt.tight_layout()
plt.savefig(save_path + 'KNN - simulated data.pdf')

if show_charts:
    plt.show()

plt.close()

# ===== DECISION REGION FOR K=1 ======
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Put the result into a color plot
plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

# Plot also the training points
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, alpha=1, edgecolor="black", palette=cmap_bold, legend=None)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
ax = plt.gca()
ax.set(xlabel='$X$', ylabel='$Y$',
       title='Classification with $K=1$')
plt.tight_layout()
plt.savefig(save_path + 'KNN - classification k=1.pdf')

if show_charts:
    plt.show()

plt.close()


# ===== CROSS VALIDATION FOR K =====
train_scores, valid_scores = validation_curve(estimator=KNeighborsClassifier(), X=X, y=y, cv=2,
                                              param_name='n_neighbors', param_range=range(1, k_test),
                                              verbose=1)

train_score = train_scores.mean(axis=1)
test_score = valid_scores.mean(axis=1)

df = pd.DataFrame(data={'Train Score': train_score,
                        'Test Score': test_score},
                  index=range(1, k_test))

K = df['Test Score'].idxmax()
print(K)

df.plot()
ax = plt.gca()
ax.set(xlabel='$K$ Nearest Neighbours', ylabel='% Correct Classifications')
ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig(save_path + 'KNN - Cross validation.pdf')

if show_charts:
    plt.show()

plt.close()


# ===== DECISION REGION FOR K=OPTIMAL ======
clf = KNeighborsClassifier(n_neighbors=K)
clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Put the result into a color plot
plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

# Plot also the training points
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, alpha=1, edgecolor="black", palette=cmap_bold, legend=None)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

ax = plt.gca()
ax.set(xlabel='$X$', ylabel='$Y$',
       title=f'Classification with $K={K}$')
plt.tight_layout()
plt.savefig(save_path + 'KNN - classification k=optimal.pdf')

if show_charts:
    plt.show()

plt.close()


# ===== DECISION REGION FOR K=50 ======
clf = KNeighborsClassifier(n_neighbors=50)
clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Put the result into a color plot
plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

# Plot also the training points
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, alpha=1, edgecolor="black", palette=cmap_bold, legend=None)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

ax = plt.gca()
ax.set(xlabel='$X$', ylabel='$Y$',
       title=f'Classification with $K=50$')
plt.tight_layout()
plt.savefig(save_path + 'KNN - classification k=50.pdf')

if show_charts:
    plt.show()

plt.close()

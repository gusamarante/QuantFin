import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve

k_test = 51
h = 0.02  # step size in the mesh

# generate the data
X, y = make_moons(noise=0.3,
                  random_state=444,
                  n_samples=200)

# plot the data
cmap_light = ListedColormap(['orange', 'cyan'])
cmap_bold = ['darkorange', 'c']

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(cmap_bold), edgecolor="black")
plt.show()

# Decision region for K=1
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Put the result into a color plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=1)

# Plot also the training points
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, alpha=1, edgecolor="black", palette=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()

# estimate for several values of K (Cross-validation)
train_scores, valid_scores = validation_curve(estimator=KNeighborsClassifier(), X=X, y=y, cv=10,
                                              param_name='n_neighbors', param_range=range(1, k_test),
                                              verbose=2)

train_score = train_scores.mean(axis=1)
test_score = valid_scores.mean(axis=1)

df = pd.DataFrame(data={'Train Score': train_score,
                        'Test Score': test_score},
                  index=range(1, k_test))

df.plot()
plt.show()

K = df['Test Score'].idxmax()
print(K)

# Decision region for K=optimal
clf = KNeighborsClassifier(n_neighbors=K)
clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Put the result into a color plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=1)

# Plot also the training points
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, alpha=1, edgecolor="black", palette=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()

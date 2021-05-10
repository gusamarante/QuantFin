import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

show_charts = False
save_path = '/Users/gustavoamarante/Dropbox/Aulas/QuantFin/figures/'

X, y = make_blobs(n_samples=1500, random_state=170)

y_pred2 = KMeans(n_clusters=2).fit_predict(X)
y_pred3 = KMeans(n_clusters=3).fit_predict(X)
y_pred4 = KMeans(n_clusters=4).fit_predict(X)
y_pred5 = KMeans(n_clusters=5).fit_predict(X)

fig, ax = plt.subplots(2, 2)
ax[0, 0].scatter(X[:, 0], X[:, 1], c=y_pred2)
ax[0, 0].set_title('$K=2$')

ax[0, 1].scatter(X[:, 0], X[:, 1], c=y_pred3)
ax[0, 1].set_title('$K=3$')

ax[1, 0].scatter(X[:, 0], X[:, 1], c=y_pred4)
ax[1, 0].set_title('$K=4$')

ax[1, 1].scatter(X[:, 0], X[:, 1], c=y_pred5)
ax[1, 1].set_title('$K=5$')

plt.suptitle('K-Means Clustering for Different Sizes')
plt.tight_layout()
plt.savefig(save_path + 'KMeans - Blobs - different sizes of K.pdf')


if show_charts:
    plt.show()

plt.close()

# Distorted Blobs
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)

y_pred2 = KMeans(n_clusters=2).fit_predict(X_aniso)
y_pred3 = KMeans(n_clusters=3).fit_predict(X_aniso)
y_pred4 = KMeans(n_clusters=4).fit_predict(X_aniso)
y_pred5 = KMeans(n_clusters=5).fit_predict(X_aniso)

fig, ax = plt.subplots(2, 2)
ax[0, 0].scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred2)
ax[0, 0].set_title('$K=2$')

ax[0, 1].scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred3)
ax[0, 1].set_title('$K=3$')

ax[1, 0].scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred4)
ax[1, 0].set_title('$K=4$')

ax[1, 1].scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred5)
ax[1, 1].set_title('$K=5$')

plt.suptitle('K-Means Clustering for Different Sizes')
plt.tight_layout()
plt.savefig(save_path + 'KMeans - distorted Blobs - different sizes of K.pdf')


if True:
    plt.show()

plt.close()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix, accuracy_score
np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
gmm = GaussianMixture(n_components=3, max_iter=100, random_state=42)
gmm.fit(X)
y_gmm = gmm.predict(X)
conf_matrix = confusion_matrix(y_true, y_gmm)
accuracy = accuracy_score(y_true, y_gmm)
print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, s=40, cmap='viridis')
plt.title('Gaussian Mixture Model (GMM) Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

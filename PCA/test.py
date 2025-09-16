import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Sample 2D data (5 samples, 2 features)
X = np.array([
    [2, 0],
    [3, 2],
    [4, 1],
    [5, 3],
    [6, 4]
])

# Center the data (important for PCA)
X_mean = X - np.mean(X, axis=0)

# Fit PCA with 2 components
pca = PCA(n_components=2)
pca.fit(X_mean)

# Get the eigenvectors (components)
components = pca.components_
explained_variance = pca.explained_variance_

# Plot the data points
plt.scatter(X_mean[:, 0], X_mean[:, 1], alpha=0.7)

# Plot eigenvectors as arrows
origin = np.mean(X_mean, axis=0)

for i in range(2):
    vector = components[i] * explained_variance[i]  # Scale by eigenvalue for visualization
    plt.arrow(origin[0], origin[1], vector[0], vector[1],
              head_width=0.3, head_length=0.3, color='red' if i == 0 else 'blue', linewidth=2)

plt.xlabel('Feature A (centered)')
plt.ylabel('Feature B (centered)')
plt.title('PCA Eigenvectors (Red=PC1, Blue=PC2)')
plt.grid(True)
plt.axis('equal')
plt.savefig('pca_eigenvectors.png')
plt.show()

import numpy as np


class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4) -> None:
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        random_indices = np.random.permutation(X.shape[0])[:self.k]
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids,ord=2, axis=2)
            self.labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[self.labels == j].mean(axis=0) for j in range(self.k)])

            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break

            self.centroids = new_centroids

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)




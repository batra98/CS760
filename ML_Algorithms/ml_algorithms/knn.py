from typing import Counter
import numpy as np

class KNN:
    def __init__(self, k=3) -> None:
        self.k = k


    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _compute_distances(self, X_test):
        """
        X_test = (n_test_samples, n_features)
        X_train = (n_train_samples, n_features)
        """

        distances = np.sqrt(np.sum(np.square(X_test[:, np.newaxis] - self.X_train), axis=2))

        return distances
    
    def _predict_single(self, distances):
        nearest_neighbours_indices = np.argsort(distances)[:self.k]

        nearest_neighbours_labels = self.y_train[nearest_neighbours_indices]

        most_common = Counter(nearest_neighbours_labels).most_common(1)

        return most_common[0][0]

    def predict(self, X_test):
        distances = self._compute_distances(X_test=X_test)

        y_pred = np.apply_along_axis(self._predict_single, 1, distances)

        return y_pred





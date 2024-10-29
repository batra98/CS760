import numpy as np


class LinearRegression:
    def __init__(self, l1=0.0, l2=0.0, learning_rate=0.001, iterations=1000, use_gradient=True) -> None:
        self.l1 = l1
        self.l2 = l2
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.use_gradient = use_gradient

    def fit(self, X, y):
        m, n = X.shape
        X_b = np.c_[np.ones((m,1)), X]

        if self.use_gradient:
            self.theta = np.zeros(n+1)

            for _ in range(self.iterations):
                y_hat = X_b @ self.theta

                error = y_hat - y

                l1_penalty = self.l1 * np.sign(self.theta)
                l2_penalty = self.l2 * self.theta

                gradient = (2/m) * X_b.T @ error + l1_penalty + 2*l2_penalty
                
                self.theta -= self.learning_rate * gradient
        else:
            identity_matrix = np.eye(n+1)
            identity_matrix[0,0] = 0
            self.theta = np.linalg.inv(X_b.T @ X_b + self.l2 * identity_matrix) @ X_b.T @ y


            if self.l1 > 0:
                self.theta = np.sign(self.theta) * np.maximum(0, np.abs(self.theta) - self.l1)

    def predict(self, X):
        m, _ = X.shape
        X_b = np.c_[np.ones((m, 1)), X]
        return X_b @ self.theta

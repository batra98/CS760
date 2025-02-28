import numpy as np
import matplotlib.pyplot as plt
from predict_label import predict


def plot_decision_boundary(root, X, Y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    
    Z = predict(root, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    
    plt.figure(figsize=(8, 6))
    Z = predict(root, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

    plt.xlabel('Feature 0 (x_0)')
    plt.ylabel('Feature 1 (x_1)')
    return plt

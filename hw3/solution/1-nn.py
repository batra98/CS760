import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

data = np.loadtxt('../data/D2z.txt')
X_train = data[:, :-1]
y_train = data[:, -1]

knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

knn.fit(X_train_scaled, y_train)

x_min, x_max = -2, 2
y_min, y_max = -2, 2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_scaled = scaler.transform(grid_points)

Z = knn.predict(grid_points_scaled)

plt.figure(figsize=(8, 6))


plt.scatter(grid_points[:, 0], grid_points[:, 1], c=Z)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o', s=100) 

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('1-NN')
plt.show()

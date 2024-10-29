"""
KNN

import numpy as np

from knn import KNN




X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
y_train = np.array([0, 1, 1, 0])

X_test = np.array([[1.5, 2.5], [3.5, 4.5]])

knn = KNN(k=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(f"Predicted labels: {y_pred}")
"""

"""
KMeans

import numpy as np

from kmeans import KMeans

X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

kmeans = KMeans(k=5)
kmeans.fit(X)


labels = kmeans.predict(X)
print(kmeans.centroids, labels)
"""



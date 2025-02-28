import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

data = pd.read_csv('../data/emails.csv', index_col=0)

print(data.columns)
X = data.drop(columns=['Prediction']).values
y = data['Prediction'].values


def cross_validation_knn(model, X, y, num_folds=5):
    fold_size = len(X) // num_folds
    metrics = {'Fold': [], 'Accuracy': [], 'Precision': [], 'Recall': []}

    for fold in range(num_folds):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size

        X_train = np.vstack([X[:test_start], X[test_end:]])
        y_train = np.hstack([y[:test_start], y[test_end:]])
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)

        metrics['Fold'].append(fold + 1)
        metrics['Accuracy'].append(accuracy)
        metrics['Precision'].append(precision)
        metrics['Recall'].append(recall)

        print(f"Fold {fold + 1}:")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n")
    
    return pd.DataFrame(metrics)



log_reg = LogisticRegression(max_iter=1000)

results = cross_validation_knn(log_reg, X, y, num_folds = 5)

print(results.to_string(index=False))

neighbors = [1,3,5,7,10]

avg_accuracy = []

for n in neighbors:
    print(f"Knn for {n} neighbors:")
    knn = KNeighborsClassifier(n_neighbors=n)
    results = cross_validation_knn(knn, X, y, num_folds=5)
    
    print(results.to_string(index = False))
    avg_accuracy.append(results['Accuracy'].mean())


plt.figure(figsize=(8, 6))


plt.plot(neighbors,avg_accuracy)

plt.xlabel('k')
plt.ylabel('Average Accuracy')
plt.title('k-NN 5 Fold Cross Validation')
plt.show()

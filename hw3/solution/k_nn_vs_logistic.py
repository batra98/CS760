import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

data = pd.read_csv('../data/emails.csv', index_col=0)

X = data.drop(columns=['Prediction']).values
y = data['Prediction'].values

X_train, X_test = X[:4000], X[4000:]
y_train, y_test = y[:4000], y[4000:]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_prob_knn = knn.predict_proba(X_test)[:, 1]
y_prob_log_reg = log_reg.predict_proba(X_test)[:, 1]

fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, y_prob_log_reg)
roc_auc_log_reg = auc(fpr_log_reg, tpr_log_reg)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, color='blue', lw=2, label=f'kNN (k=5) ROC curve (area = {roc_auc_knn:.2f})')
plt.plot(fpr_log_reg, tpr_log_reg, color='red', lw=2, label=f'Logistic Regression ROC curve (area = {roc_auc_log_reg:.2f})')


plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('kNN vs Logistic Regression')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f'kNN (k=5) ROC AUC: {roc_auc_knn:.2f}')
print(f'Logistic Regression ROC AUC: {roc_auc_log_reg:.2f}')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


confidence_scores = np.array([0.95, 0.85, 0.8, 0.7, 0.55, 0.45, 0.4, 0.3, 0.2, 0.1])
true_labels = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 0])


fpr, tpr, thresholds = roc_curve(true_labels, confidence_scores, pos_label=1)


roc_auc = auc(fpr, tpr)


plt.figure()
plt.plot(fpr, tpr, color='red', lw=2)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.show()

import numpy as np

def predict_single(node, x):
    
    if node.label is not None:
        return node.label

    
    if x[node.feature] >= node.threshold:
        return predict_single(node.left, x)
    else:
        return predict_single(node.right, x)

def predict(tree, X):
    
    if len(X.shape) == 1:  
      X = X.reshape(1, -1)
    return np.array([predict_single(tree, x) for x in X])

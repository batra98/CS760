import numpy as np
from math import log2

class DecisionTreeNode:

    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label


def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum([p * log2(p) for p in probabilities if p > 0])

def information_gain(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] >= threshold
    right_mask = ~left_mask

    parent_entropy = entropy(y)

    left_entropy = entropy(y[left_mask])
    right_entropy = entropy(y[right_mask])

    #print(f"{threshold} & left_entropy={left_entropy} & right_entropy={right_entropy}\\\\")
    n = len(y)
    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)

    weighted_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy

    ig = parent_entropy - weighted_entropy

    return ig

def information_gain_ratio(X, y, feature_index, threshold):

    ig = information_gain(X, y, feature_index, threshold)

    #print(f"{threshold} & ig={ig}\\\\")

    left_mask = X[:, feature_index] >= threshold
    right_mask = ~left_mask

    n = len(y)
    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)


    split_info = -((n_left / n) * log2(n_left / n) + (n_right / n) * log2(n_right / n)) if n_left > 0 and n_right > 0 else 0
    #print(f"{threshold} & split_info={split_info}\\\\")

    return ig / split_info if split_info != 0 else 0
def best_split(X, y):
    best_feature = None
    best_threshold = None
    best_gain_ratio = -float('inf')

    n_features = X.shape[1]

    for feature_index in range(n_features):


        unique_values = np.unique(X[:, feature_index])


        for threshold in unique_values:

            gain_ratio = information_gain_ratio(X, y, feature_index, threshold)


            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_feature = feature_index
                best_threshold = threshold

                # print(f"{best_gain_ratio} & {best_feature} & {best_threshold}\\\\")

    return best_feature, best_threshold, best_gain_ratio

def majority_class(y):

    unique_classes, counts = np.unique(y, return_counts=True)
    majority_label = unique_classes[np.argmax(counts)]
    return majority_label if len(unique_classes) == 1 or counts[0] != counts[1] else 1

def build_tree(X, y, min_samples_split=2):
    if len(y) == 0:
        return DecisionTreeNode(label=1)

    if len(np.unique(y)) == 1:
        return DecisionTreeNode(label=y[0])

    if len(y) < min_samples_split:
        return DecisionTreeNode(label=majority_class(y))

    best_feature, best_threshold, best_gain_ratio = best_split(X, y)




    if best_gain_ratio == 0:
        return DecisionTreeNode(label=majority_class(y))

    left_mask = X[:, best_feature] >= best_threshold
    right_mask = ~left_mask

    left_subtree = build_tree(X[left_mask], y[left_mask])
    right_subtree = build_tree(X[right_mask], y[right_mask])

    return DecisionTreeNode(
        feature=best_feature,
        threshold=best_threshold,
        left=left_subtree,
        right=right_subtree
    )

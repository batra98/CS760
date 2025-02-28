from decision_tree import build_tree
from predict_label import predict
from plot_decision_boundary import plot_decision_boundary
from load_data import load_data
from print_tree import print_tree, visualize_tree
from sklearn.metrics import accuracy_score
import numpy as np
from count_nodes_of_tree import count_nodes
import matplotlib.pyplot as plt
"""1.3
X,Y = load_data("../../hw2/data/Druns.txt")
print(X.shape,Y.shape)

root = build_tree(X, Y)
"""

"""1.4
X,Y = load_data("../../hw2/data/D3leaves.txt")
print(X.shape,Y.shape)

root = build_tree(X, Y)
print_tree(root)
visualize_tree(root).render("decision_tree-D3leaves", format='png')
"""


"""1.5
X,Y = load_data("../../hw2/data/D2.txt")
print(X.shape,Y.shape)
tree = build_tree(X, Y)


dot = visualize_tree(tree)
dot.render('decision_tree-D2', format='png')
"""

"""1.6
X,Y = load_data("../../hw2/data/D2.txt")
print(X.shape,Y.shape)

root = build_tree(X, Y)

plt = plot_decision_boundary(root, X,Y)
plt.show()
"""

"""1.7
data = np.loadtxt("../../hw2/data/D2.txt")
X = data[:, :-1]
Y = data[:, -1]

print(X,Y)
np.random.seed(42)
indices = np.random.permutation(data.shape[0])


train_size = 8192
train_indices = indices[:train_size]
test_indices = indices[train_size:]


candidate_training_set = data[train_indices]
test_set = data[test_indices]
X_test = test_set[:, :-1]
Y_test = test_set[:, -1]


sizes = [32, 128, 512, 2048, 8192]


num_nodes = []
test_errors = []


x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))


for n in sizes:
    
    X_train = candidate_training_set[:n, :-1]
    Y_train = candidate_training_set[:n, -1]

    
    root = build_tree(X_train, Y_train)

    
    Y_pred = predict(root, X_test)
    test_error = 1 - accuracy_score(Y_test, Y_pred)
    test_errors.append(test_error)
    print(test_error)
    
    num_nodes.append(count_nodes(root))

    
    plt.figure(figsize=(8, 6))
    Z = predict(root, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    
    plt.title(f'Decision Boundary for Training Size: {n}')
    plt.xlabel('Feature 0 (x_0)')
    plt.ylabel('Feature 1 (x_1)')
    plt.show()


plt.figure(figsize=(10, 6))
plt.plot(sizes, test_errors, marker='o')
plt.xscale('log')
plt.xlabel('Training Set Size (n)')
plt.ylabel('Test Error (errn)')
plt.title('Learning Curve: Test Error vs. Training Set Size')
plt.grid()
plt.show()


for n, nodes, error in zip(sizes, num_nodes, test_errors):
    print(f'n: {n}, Number of Nodes: {nodes}, Test Error: {error:.4f}')

"""


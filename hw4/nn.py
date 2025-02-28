import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, torch, transforms
from torch.utils.data import DataLoader

input_size = 784 
hidden_size1 = 300
hidden_size2 = 200
num_classes = 10
learning_rate = 0.01
batch_size = 64
num_epochs = 5

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class SimpleNN:
    def __init__(self):
        self.W1 = np.random.randn(hidden_size1, input_size) * 0.01
        self.W2 = np.random.randn(hidden_size2, hidden_size1) * 0.01
        self.W3 = np.random.randn(num_classes, hidden_size2) * 0.01

    def forward(self, x):
        self.z1 = np.dot(x, self.W1.T)
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2.T)
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W3.T)
        self.y_hat = softmax(self.z3)
        return self.y_hat

    def backward(self, x, y):
        m = y.shape[0]
        
        dL_dy_hat = self.y_hat - y
        dL_dW3 = np.dot(dL_dy_hat.T, self.a2) / m
        
        dL_da2 = np.dot(dL_dy_hat, self.W3)
        dL_dz2 = dL_da2 * sigmoid_derivative(self.z2)
        dL_dW2 = np.dot(dL_dz2.T, self.a1) / m
        
        dL_da1 = np.dot(dL_dz2, self.W2)
        dL_dz1 = dL_da1 * sigmoid_derivative(self.z1)
        dL_dW1 = np.dot(dL_dz1.T, x) / m
        
        self.W1 -= learning_rate * dL_dW1
        self.W2 -= learning_rate * dL_dW2
        self.W3 -= learning_rate * dL_dW3

    def train(self, train_loader):
        for x, y in train_loader:
            y_onehot = np.zeros((y.size(0), num_classes))
            y_onehot[np.arange(y.size(0)), y.numpy()] = 1
            _ = self.forward(x.numpy())
            self.backward(x.numpy(), y_onehot)

    def evaluate(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                y_hat = self.forward(x.numpy())
                predicted = np.argmax(y_hat, axis=1)
                total += y.size(0)
                correct += (predicted == y.numpy()).sum()
        return correct / total

model = SimpleNN()
train_errors = []
test_errors = []

for epoch in range(num_epochs):
    model.train(train_loader)
    train_accuracy = model.evaluate(train_loader)
    test_accuracy = model.evaluate(test_loader)
    train_errors.append(1 - train_accuracy)
    test_errors.append(1 - test_accuracy)
    print(f'Epoch {epoch+1}, Train Error: {train_errors[-1]}, Test Error: {test_errors[-1]}')

plt.plot(range(num_epochs), train_errors, label='Train Error')
plt.plot(range(num_epochs), test_errors, label='Test Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.title('Learning Curve')
plt.show()

import numpy as np

def load_data(file_path):
    data = np.loadtxt(file_path)
    X = data[:, :-1]
    Y = data[:, -1]
    return X,Y


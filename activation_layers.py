import numpy as np

def activation(x, f):
    return f(x)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

# Sigmoid dash function (used for backward prop)
def sigmoid_dash(x):
    sig = sigmoid(x)
    return sig * (1 - sig)
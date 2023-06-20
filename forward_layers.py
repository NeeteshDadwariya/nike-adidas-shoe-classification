import numpy as np

from activation_layers import sigmoid

"""
Forward layer
"""


def preforward(x, w, b):
    return np.dot(x, w) + b


"""
Prediction function
"""


def predict(x, W1, B1, W2, B2):
    forward_op1 = preforward(x, W1, B1)
    activation_op1 = sigmoid(forward_op1)
    forward_op2 = preforward(activation_op1, W2, B2)
    prediction = sigmoid(forward_op2)
    return prediction

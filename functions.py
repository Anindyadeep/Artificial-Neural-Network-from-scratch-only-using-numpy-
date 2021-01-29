import numpy as np


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def sigmoid_derivative(dA, Z):
    S = sigmoid(Z)
    return dA * S * (1 - S)


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

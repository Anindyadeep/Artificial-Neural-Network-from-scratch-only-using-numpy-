import numpy as np


def binary_loss(A, Y):
    m = A.shape[1]
    cost = -1 / m * np.sum(Y * np.log(A + 1e-8) + (1 - Y) * (np.log(1 - A + 1e-8)))
    return cost


def accuracy(A, Y):
    m = A.shape[1]
    acc = (np.sum(np.round(A) == Y)) / m
    return acc

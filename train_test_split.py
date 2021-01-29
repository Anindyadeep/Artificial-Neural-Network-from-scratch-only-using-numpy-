import numpy as np


def train_dev_test_split(X, Y, split_percentage):
    if int(X.shape[0]) > int(X.shape[1]):

        Y = Y.reshape(X.shape[0], 1)
        split = X.shape[0] - int(X.shape[0] * split_percentage)
        X_train = np.array(X[:split, :])
        Y_train = np.array(Y[:split, :])
        X_test = np.array(X[split:, :])
        Y_test = np.array(Y[split:, :])

        X_test, X_val = np.array_split(X_test, 2)
        Y_test, Y_val = np.array_split(Y_test, 2)

        X_train, Y_train, X_val, Y_val, X_test, Y_test = X_train.T, Y_train.T, X_val.T, Y_val.T, X_test.T, Y_test.T
    else:
        Y = Y.reshape(1, X.shape[1])
        split = X.shape[1] - int(X.shape[1] * split_percentage)
        X_train = np.array(X[:, :split])
        Y_train = np.array(Y[:, :split])
        X_test = np.array(X[:, split:])
        Y_test = np.array(Y[:, split:])

        X_train, Y_train = X_train.T, Y_train.T
        X_test, Y_test = X_test.T, Y_test.T

        X_test, X_val = np.array_split(X_test, 2)
        Y_test, Y_val = np.array_split(Y_test, 2)
        X_train, Y_train, X_val, Y_val, X_test, Y_test = X_train.T, Y_train.T, X_val.T, Y_val.T, X_test.T, Y_test.T

    return X_train, Y_train, X_val, Y_val, X_test, Y_test



from functions import *


def single_feed_forward(A, W, b, activation="relu"):
    Z = np.dot(W, A) + b
    if activation == 'relu':
        activation = relu
    else:
        activation = sigmoid
    A = activation(Z)
    return Z, A


def full_feed_forward(X, params):
    cache = {}

    cache["A0"] = X
    for i in range(1, params["layers"]):
        A_prev = cache["A" + str(i - 1)]
        W_curr = params["W" + str(i)]
        b_curr = params["b" + str(i)]
        activation_curr = params["act" + str(i)]

        Z_curr, A_curr = single_feed_forward(A_prev, W_curr, b_curr, activation_curr)
        cache["A" + str(i)] = A_curr
        cache["Z" + str(i)] = Z_curr

    AL = A_curr
    return AL, cache



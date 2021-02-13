from functions import *
from forward import *
from cost_acc import *

def single_layer_back_prop(dA_curr, Z_curr, A_prev, W_curr, activation='relu'):
    m = A_prev.shape[1]

    if activation == 'relu':
        activation_backward = relu_derivative
    else:
        activation_backward = sigmoid_derivative

    dZ_curr = activation_backward(dA_curr, Z_curr)
    dW_curr = 1 / m * (np.dot(dZ_curr, A_prev.T))
    db_curr = 1 / m * (np.sum(dZ_curr, axis=1, keepdims=True))
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dW_curr, db_curr, dA_prev


def compute_grads(cache, params, Y):
    grads = {}

    layers = params["layers"]
    l = layers - 1
    Y_hat = cache["A" + str(l)]

    dA_prev = -(np.divide(Y, Y_hat + 1e-8) - np.divide(1 - Y, 1 - Y_hat))

    while l >= 1:
        dA_curr = dA_prev  # in order to make a match with the order in the loop

        Z_curr = cache["Z" + str(l)]
        A_prev = cache["A" + str(l - 1)]
        W_curr = params["W" + str(l)]
        activation = params["act" + str(l)]

        dW_curr, db_curr, dA_prev = single_layer_back_prop(dA_curr, Z_curr, A_prev, W_curr, activation)

        grads["dW" + str(l)] = dW_curr
        grads["db" + str(l)] = db_curr

        l -= 1

    return grads


def update_params(params, grads, learning_rate):
    layers = params["layers"]
    for l in range(1, layers):
        params["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        params["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return params


# test part 3

def train(back_prop):
    X_train = back_prop["X_train"]
    Y_train = back_prop["Y_train"]

    X_val = back_prop["X_val"]
    Y_val = back_prop["Y_val"]

    params = back_prop["params"]
    epochs = back_prop["epochs"]
    learning_rate = back_prop["learning_rate"]

    LOSS_TRAIN = []
    ACC_TRAIN = []

    LOSS_VAL = []
    ACC_VAL = []

    for epoch in range(epochs):
        AL_train, cache_train = full_feed_forward(X_train, params)
        AL_val, cache_val = full_feed_forward(X_val, params)

        grads = compute_grads(cache_train, params, Y_train)
        params = update_params(params, grads, learning_rate)

        loss_train = binary_loss(AL_train, Y_train)
        acc_train = accuracy(AL_train, Y_train)

        loss_val = binary_loss(AL_val, Y_val)
        acc_val = accuracy(AL_val, Y_val)

        LOSS_TRAIN.append(loss_train)
        ACC_TRAIN.append(acc_train)

        LOSS_VAL.append(loss_val)
        ACC_VAL.append(acc_val)

        if epoch % 100 == 0:
            print("AT EPOCH: ", epoch, " TRAIN LOSS: ", loss_train, "TRAIN ACC: ", acc_train)
            print("AT EPOCH: ", epoch, " VAL LOSS: ", loss_val, "VAL ACC: ", acc_val)
            print("\n")

    return params, LOSS_TRAIN, ACC_TRAIN, LOSS_VAL, ACC_VAL

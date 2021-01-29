from architecture import *
from custom_data_non_img import *
import matplotlib.pyplot as plt
from backward import *

path = r'C:\Users\cosmi\Documents\Datasets\Breast Cancer\data.csv'
X_train, Y_train, X_val, Y_val, X_test, Y_test = get_breast_cancer(path)

nodes = [X_train.shape[0],16, 32, 16, 4, 4, 1]
act_func = ['relu', 'relu','relu', 'relu', 'relu', 'sigmoid']
params = get_params(nodes, act_func)

AL, cache = full_feed_forward(X_train, params)
print("CURRENT LOSS BEFORE TRAIN: ", binary_loss(AL, Y_train))
print("CURRENT ACC BEFORE TRAIN: ", accuracy(AL, Y_train))
print('\n')

back_prop = {
    "X_train": X_train,
    "Y_train": Y_train,
    "X_val" : X_val,
    "Y_val": Y_val,
    "learning_rate": 0.003,
    "epochs": 1300,
    "params": params
    }

params, LOSS_TRAIN, ACC_TRAIN, LOSS_VAL, ACC_VAL = train(back_prop)

A_test, cache_test = full_feed_forward(X_test, params)

print("CURRENT LOSS TEST: ", binary_loss(A_test, Y_test))
print("CURRENT ACC TEST: ", accuracy(A_test, Y_test))


plt.plot(LOSS_TRAIN)
plt.plot(LOSS_VAL)
#plt.plot(ACC_TRAIN)
#plt.plot(ACC_VAL)
import pandas as pd
import numpy as np
from train_test_split import train_dev_test_split
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons


############################## PREPARE THE TITANIC DATASET ##############################

def organise_titanic(data):
    data_req = {'Pclass': data['Pclass'],
                'Sex': data['Sex'],
                'Age': data['Age'],
                'Fare': data['Fare'],
                'Embarked': data['Embarked'],
                'Survived': data['Survived']
                }

    data_mod = pd.DataFrame(data_req)

    sex_encoded = pd.get_dummies(data_mod.Sex)
    embarked_encoded = pd.get_dummies(data_mod.Embarked)
    pclass_encoded = pd.get_dummies(data_mod.Pclass, prefix='class')

    data_mod = data_mod.drop('Sex', axis=1)
    data_mod = data_mod.drop('Embarked', axis=1)
    data_mod = data_mod.drop('Pclass', axis=1)

    data_mod = data_mod.join(sex_encoded)
    data_mod = data_mod.join(embarked_encoded)
    data_mod = data_mod.join(pclass_encoded)

    data_mod = data_mod.drop('male', axis=1)

    data_mod['Fare'] = (data_mod['Fare'] - np.mean(data_mod['Fare'])) / np.std(data_mod['Fare'])
    data_mod['Age'] = (data_mod['Age'] - np.mean(data_mod['Age'])) / np.std(data_mod['Age'])

    data_mod = data_mod.dropna()

    return data_mod


def generate_titanic(path_train, path_test, path_val):
    train_data = pd.read_csv(path_train)
    test_data = pd.read_csv(path_test)
    test_data_val = pd.read_csv(path_val)

    val = test_data_val.iloc[:, 1:]
    val_data = test_data.join(val)
    train_data_mod = organise_titanic(train_data)
    test_data_mod = organise_titanic(val_data)

    X_train = np.array(train_data_mod.iloc[:, :-1])
    Y_train = np.array(train_data_mod.iloc[:, -1:])

    X_test = np.array(test_data_mod.iloc[:, :-1])
    Y_test = np.array(test_data_mod.iloc[:, -1:])

    X = np.concatenate([X_train, X_test], axis=0)
    Y = np.concatenate([Y_train, Y_test], axis=0)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = train_dev_test_split(X, Y, 0.3)

    X_train = (X_train - (np.mean(X_train))) / np.std(X_train)
    X_val = (X_val - (np.mean(X_val))) / np.std(X_val)
    X_test = (X_test - (np.mean(X_test))) / np.std(X_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


############################## PREPARE THE CIRCLE DATASET ##############################

def get_circle_data():
    feature, labels = make_circles(n_samples=10000, noise=0.01)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = train_dev_test_split(feature, labels, 0.3)

    X_train = (X_train - (np.mean(X_train))) / np.std(X_train)
    X_val = (X_val - (np.mean(X_val))) / np.std(X_val)
    X_test = (X_test - (np.mean(X_test))) / np.std(X_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


############################## PREPARE THE MOONs DATASET ##############################

def get_moons_data():
    feature, labels = make_moons(n_samples=10000, noise=0.01)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = train_dev_test_split(feature, labels, 0.3)

    X_train = (X_train - (np.mean(X_train))) / np.std(X_train)
    X_val = (X_val - (np.mean(X_val))) / np.std(X_val)
    X_test = (X_test - (np.mean(X_test))) / np.std(X_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


############################## PREPARE THE BREAST CANCER DATASET ##############################

def get_breast_cancer(path):
    data = pd.read_csv(path)
    data_label = pd.get_dummies(data.diagnosis)
    data_mod = data.iloc[:, 1:-1].join(data_label)
    data_mod = data_mod.drop('diagnosis', axis=1)
    data_mod = data_mod.drop('B', axis=1)
    X = np.array(data_mod.iloc[:, :-1])
    Y = np.array(data_mod.iloc[:, -1:])
    X = (X - np.mean(X)) / np.std(X)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = train_dev_test_split(X, Y, 0.3)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


############################## PREPARE THE FLOWER DATASET (TAKEN FROM ANDREW NG) ##############################

def get_flower(m):
    np.random.seed(1)
    N = int(m / 2)
    D = 2
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype='uint8')
    a = 2
    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.14, (j + 1) * 3.14, N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.1
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    X_train, Y_train, X_val, Y_val, X_test, Y_test = train_dev_test_split(X, Y, 0.3)
    X_train = (X_train - (np.mean(X_train))) / np.std(X_train)
    X_val = (X_val - (np.mean(X_val))) / np.std(X_val)
    X_test = (X_test - (np.mean(X_test))) / np.std(X_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

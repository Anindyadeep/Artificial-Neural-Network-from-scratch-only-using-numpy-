import numpy as np


def get_params(nodes, act_func):
    params = {}
    layers = len(nodes)
    params["layers"] = layers
    for i in range(1, layers):
        params["W" + str(i)] = np.random.randn(nodes[i], nodes[i - 1]) * np.sqrt(2 / nodes[i])
        params["b" + str(i)] = np.zeros(shape=(nodes[i], 1))
        params["act" + str(i)] = act_func[i - 1]

    return params

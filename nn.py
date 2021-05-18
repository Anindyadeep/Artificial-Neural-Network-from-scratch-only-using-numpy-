import numpy as np

class neural_network:
    '''
    making a class of the neural network which will work
    for now as a simple numpy ANN
    '''
    def __init__(self, Input):
        self._layers = 1
        self._neurons = []
        self._activation_functions = []
        self._parameters = {}
        self.input = Input
        self._neurons.append(self.input.shape[0])
        self._cache = {}

    ########### ADDING THE NEURONS AND MAKING A DYNAMIC CONNECTION ALONGSIDE ###########

    def add_neurons(self, num_neurons, activation = "relu"):
        self._neurons.append(num_neurons)
        self._activation_functions.append(activation)
        self._layers +=1

        if self._layers >= 2:
            self._parameters["w"+str(self._layers-1)] = np.random.randn(self._neurons[self._layers-1], self._neurons[self._layers-2])
            self._parameters["b"+str(self._layers-1)] = np.zeros(shape=(self._neurons[self._layers-1], 1))
            self._parameters["act"+str(self._layers-1)] = self._activation_functions[self._layers-2]

    ############ ACTIVATION PART #############
    
    def _relu(self, Z):
        return np.maximum(0, Z)
    
    def _relu_derivative(self, dA, Z):
         dZ = np.array(dA, copy=True)
         dZ[Z <= 0] = 0
         return dZ

    def _sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A

    def _sigmoid_derivative(self, dA, Z):
        S = self._sigmoid(Z)
        return dA * S * (1 - S)

    ############ THE FEED FORWARD FUCNTION #############

    def _single_feed_forward(self, A, W, b, activation = 'relu'):
        Z = np.dot(W, A)+b
        if activation == 'relu':
            neural_activation = self._relu
        else:
            neural_activation = self._sigmoid
        A = neural_activation(Z)
        return Z, A
    
    def _full_feed_forward_function(self, X):
        self._cache["A0"] = X
        for i in range(1, self._layers):
            A_prev = self._cache["A"+str(i-1)]
            W_curr = self._parameters["w"+str(i)]
            b_curr = self._parameters["b"+str(i)]
            Z_curr, A_curr = self._single_feed_forward(A_prev, W_curr, b_curr, self._activation_functions[i-1])
            self._cache["A"+str(i)] = A_curr
            self._cache["Z"+str(i)] = Z_curr
        AL = A_curr
        return AL
    
    def get_cache(self):
        AL = self._full_feed_forward_function(self.input)
        return self._cache, AL

    ############ GETTING INFO PART #############
    def get_info(self):
        print("\nTOTAL NUMBER OF LAYERS: ", self._layers)
        print("THE INPUT LAYER NEURONS: ", self._neurons[0], "\n")
        print("---------------- TAKING THE LAYER NUMBER AS 0 BASED INDEXING ----------------")

        for i in range(1, self._layers):

            print("AT LAYER NUMBER: ", i)
            print("THE NUMBER OF NEURONS PRESENT: ", self._neurons[i])
            print("THE ACTIVATION FUNCTIONS ALLOCATED: ", self._activation_functions[i-1])
            print("THE SHAPE OF THE WEIGHT MATRIX IS: ", self._parameters["w"+str(i)].shape)
            print("THE SHAPE OF THE BIAS VECTOR IS:   ", self._parameters["b"+str(i)].shape)
            print("##############################################\n")


Input = np.random.randn(16, 10)

nn = neural_network(Input)
nn.add_neurons(8, "relu")
nn.add_neurons(7, "relu")
nn.add_neurons(1, "sigmoid")

nn.get_info()
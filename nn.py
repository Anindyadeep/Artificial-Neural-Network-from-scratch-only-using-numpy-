import numpy as np

class neural_network:
    '''
    making a class of the neural network which will work
    for now as a simple numpy ANN
    '''
    def __init__(self, Input, Label, epochs, learning_rate):
        self._layers = 1
        self._neurons = []
        self._activation_functions = []
        self._parameters = {}
        self._cache = {}
        self._grads = {}
        self._LOSS_TRAIN = []
        self._ACC_TRAIN = []

        self._LOSS_VAL = []
        self._ACC_VAL = []

        # get public input
        self.input = Input
        self.label = Label
        self.epochs = epochs
        self.learning_rate = learning_rate
        self._neurons.append(self.input.shape[0])

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

    ############ GETTING THE LOGITS #############

    def _binary_cross_entropy(self, A, Y):
        number_example = A.shape[1]
        cost = -1 / number_example * np.sum(Y * np.log(A + 1e-8) + (1 - Y) * (np.log(1 - A + 1e-8)))
        return cost
    
    def accuracy(self, A, Y_actual):
        number_example = A.shape[1]
        accuracy = (np.sum(np.round(A) == Y_actual)) / number_example
        return accuracy
    
    ############ BACKWARD PROPAGATION #############

    # single layer backpropagation
    def _single_layer_backward_propagation(self, dA_curr, Z_curr, A_prev, W_curr, activation='relu'):

        num_examples = A_prev.shape[1]
        if activation == 'relu':
            activation_backward = self._relu_derivative
        else:
            activation_backward = self._sigmoid_derivative
        
        dZ_curr = activation_backward(dA_curr, Z_curr)
        dW_curr = 1 / num_examples * (np.dot(dZ_curr, A_prev.T))
        db_curr = 1 / num_examples * (np.sum(dZ_curr, axis=1, keepdims=True))
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dW_curr, db_curr, dA_prev
    
    # computing gradients in the backpropagation
    def _compute_grads(self,Y_true):
        layers = self._layers
        l = layers-1
        Y_hat = self._cache["A" + str(l)]
        dA_prev = -(np.divide(Y_true, Y_hat + 1e-8) - np.divide(1 - Y_true, 1 - Y_hat))

        while l>=1:
            dA_curr = dA_prev  # in order to make a match with the order in the loop
            Z_curr = self._cache["Z" + str(l)]
            A_prev = self._cache["A" + str(l - 1)]
            W_curr = self._parameters["w" + str(l)]
            activation = self._parameters["act" + str(l)]
            dW_curr, db_curr, dA_prev = self._single_layer_backward_propagation(dA_curr, Z_curr, A_prev, W_curr, activation)

            self._grads["dW" + str(l)] = dW_curr
            self._grads["db" + str(l)] = db_curr

            l-=1    

    # updating the parameters in backpropagation     
    def _update_parameters(self):
        layers = self._layers
        for l in range(1, layers):
            self._parameters["w"+str(l)] -= self.learning_rate * self._grads["dW"+str(l)]
            self._parameters["b"+str(l)] -= self.learning_rate * self._grads["db"+str(l)]
    
    # full backward propagation
    def _train(self):
        
        ratio = int(len(self.input)*0.2)
        X_train = self.input[ratio:]
        Y_train = self.label[ratio:]

        X_val = self.input[:ratio]
        Y_val = self.label[:ratio]

        epochs = self.epochs

        for epoch in range(epochs):
            AL_train = self._full_feed_forward_function(X_train)
            AL_val = self._full_feed_forward_function(X_val)

            self._grads = self._compute_grads(Y_true=Y_train)
            self._parameters = self._update_parameters()

            loss_train = self._binary_cross_entropy(AL_train, Y_train)
            acc_train = self.accuracy(AL_train, Y_train)

            loss_val = self._binary_cross_entropy(AL_val, Y_val)
            acc_val = self.accuracy(AL_val, Y_val)

            self._LOSS_TRAIN.append(loss_train)
            self._ACC_TRAIN.append(acc_train)

            self._LOSS_VAL.append(loss_val)
            self._ACC_VAL.append(acc_val)

            if epoch % 100 == 0:
                print("AT EPOCH: ", epoch, " TRAIN LOSS: ", loss_train, "TRAIN ACC: ", acc_train)
                print("AT EPOCH: ", epoch, " VAL LOSS: ", loss_val, "VAL ACC: ", acc_val)
                print("\n")

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
    
    def get_history(self):
        self._train()



Input = np.random.randn(100, 4)

Label = np.array([0,1,1,0]*25)
Label = Label.reshape(100,1)

nn = neural_network(Input, Label, 500, 0.003)
nn.add_neurons(4, "relu")
nn.add_neurons(2, "relu")
nn.add_neurons(1, "sigmoid")

nn.get_info()
print(nn.get_history())

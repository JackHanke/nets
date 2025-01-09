import json
import random
import numpy as np
from time import time
import matplotlib.pyplot as plt 
from functions.activation_funcs import Identity

# example train script for ANN
def train_ann(model, train_data, train_labels, valid_data, valid_labels, batch_size, epochs, optimizer, verbose=False, plot_learning=False, N=None):
    train_cost_history, valid_cost_history = [], []
    if N is not None: N = train_data.shape[1]
    for epoch in range(epochs):
        start = time()
        # TODO make this stochastic 
        for batch_index in range(train_data.shape[1]//batch_size):
            train_data_batch = train_data[:, range(batch_index*batch_size, ((batch_index+1)*batch_size))]
            labels_batch = train_labels[:, range(batch_index*batch_size, ((batch_index+1)*batch_size))]

            train_cost, delta = model._backward(
                activation = train_data_batch,
                label = labels_batch,
                N = N
            )
            
            optimizer.step(
                weights = model.weights, 
                weights_gradients = model.weights_gradients,
                biases = model.biases,
                biases_gradients = model.biases_gradients
            )

        train_cost_history.append(train_cost)
        end = time()
        if valid_data is not None:
            # validation performance
            validation_inferences = model._forward(activation=valid_data)
            validation_cost = model.loss.cost(validation_inferences, valid_labels)
            valid_cost_history.append(validation_cost)
        if verbose and (epoch % 2) == 0: 
            print(f'Training cost after epoch {epoch} = {train_cost:.6f}. Completed in {end-start:.4f}s') 
            if valid_data is not None: print(f'Validation cost after epoch {epoch} = {validation_cost:.6f}') 
    
    if plot_learning: # plot learning curves
        plt.plot([i for i in range(1, epochs+1)], train_cost_history, label=f'Train')
        if valid_data is not None:  plt.plot([i for i in range(1, epochs+1)], valid_cost_history, label=f'Validation')
        plt.title(f'Training and validation cost per epoch')
        plt.legend(loc='upper right')
        plt.xlabel(f'Epoch')
        plt.ylabel(f'Cost (MSE)')
        plt.show()

# example test script for ANN 
def test_ann(model, test_data, test_labels, verbose=False):
    results_vector = model.inference(data=test_data)
    correct_inferences = np.equal(results_vector, test_labels).sum()
    total_inferences = (test_labels.shape[1])
    if verbose: print(f'Correct inferences={correct_inferences} out of {total_inferences} total inferences.')
    return correct_inferences/total_inferences

# Custom ANN class
#   this code follows the notation from the textbook http://neuralnetworksanddeeplearning.com/
class ArtificialNeuralNetwork:
    # dims is tuple of length >=2 that defines the model dimensions
    #   ie. (784, 15, 10) means a 784 x 15 array and a 15 x 10 array 
    # activations is tuple of activation function objects
    def __init__(self, dims, activation_funcs, loss, seed=None, version_num=0):
        self.version_num = str(version_num)
        self.dims = dims
        if len(dims)-1 != len(activation_funcs): raise Exception("List of dimensions and activations do not match.")
        self.num_layers = len(dims)
        self.loss = loss
        if seed is not None: np.random.seed(seed)
        self.activation_funcs = [-1, Identity()] + activation_funcs # insert filler to align indexing with textbook
        self.weights = [-1,-1] # insert filler to align indexing with textbook
        self.weights_gradients = [-1 for _ in range(len(dims)+1)]
        self.biases = [-1,-1]
        self.biases_gradients = [-1 for _ in range(len(dims)+1)]
        for dim_index in range(len(dims)-1):
            self.weights.append(np.random.normal(loc=0, scale=1/np.sqrt(dims[0]), size=(dims[dim_index+1], dims[dim_index])))
            self.biases.append(np.random.normal(loc=0, scale=1, size=(dims[dim_index+1], 1)))

    # forward pass
    def _forward(self, activation, include=False):
        weighted_inputs = [-1, activation]
        activations = [-1, activation]
        for layer_index in range(2,self.num_layers+1):
            weighted_input = np.dot(self.weights[layer_index], activation) + \
                            np.dot(self.biases[layer_index], np.ones((1,activation.shape[1])))
            weighted_inputs.append(weighted_input)
            activation = self.activation_funcs[layer_index].function(weighted_input)
            activations.append(activation)
        if include: return activation, weighted_inputs, activations
        else: return activation

    # forward and backward pass
    def _backward(self, activation, label, N=None, epsilon=None):
        # forward pass
        activation, weighted_inputs, activations = self._forward(activation, include=True)
        # compute cost of forward pass for verbose output
        cost = self.loss.cost(activation, label)
        # backward pass, starting with final layer
        delta = np.multiply(self.loss.loss_prime(activation, label, epsilon=epsilon), self.activation_funcs[-1].function_prime(weighted_inputs[-1]))
        #remaining layers
        for layer_index in range(self.num_layers, 1, -1):
            # compute product before weights change
            product = np.dot(self.weights[layer_index].transpose(), delta)
            m = activations[layer_index-1].shape[1] # batch_size
            weight_gradient = (np.dot(delta, activations[layer_index-1].transpose()))*(1/m)
            bias_gradient = (delta).mean(axis=1, keepdims=True)
            # add computed gradients
            self.weights_gradients[layer_index] = weight_gradient
            self.biases_gradients[layer_index] = bias_gradient
            # computes (layer_index - 1) delta vector
            # NOTE this computes first layer delta if the ann is pipelines from another model
            delta = np.multiply(product, self.activation_funcs[layer_index-1].function_prime(weighted_inputs[layer_index-1]))
        return cost, delta

    # conduct an inference for a one-hot prediction task
    def inference(self, data):
        return np.argmax(self._forward(data), axis=0, keepdims=True)

    # get the number of parameters
    def num_params(self):
        num_parameters = 0
        for weight_matrix in self.weights[2:]:
            dims = weight_matrix.shape
            num_parameters += (dims[0] * (dims[1]+1)) # +1 for biases 
        return num_parameters

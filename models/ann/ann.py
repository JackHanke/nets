# this code closely follows the textbook http://neuralnetworksanddeeplearning.com/
import numpy as np
import random
from math import sqrt
import json
import matplotlib.pyplot as plt

class ArtificialNeuralNetwork:
    # dims is tuple of length >=2 that defines the model dimensions
    #   ie. (784, 15, 10) means a 784 x 15 array and a 15 x 10 array 
    # activations is tuple of activation function objects
    # loss is a tuple of a loss function and its derivative that accepts an activation vector and label vector
    def __init__(self, dims, activation_funcs, loss, seed=None, version_num=0):
        self.version_num = str(version_num)
        if len(dims)-1 != len(activation_funcs): raise Exception("List of dimensions and activations do not match.")
        self.num_layers = len(dims)
        self.loss = loss
        if seed is not None: np.random.seed(seed)
        self.activation_funcs = [-1,-1] + activation_funcs # insert filler to align indexing with textbook
        self.weights = [-1,-1] # insert filler to align indexing with textbook
        self.biases = [-1,-1]
        for dim_index in range(len(dims)-1):
            self.weights.append(np.random.normal(loc=0, scale=1/sqrt(dims[0]), size=(dims[dim_index+1], dims[dim_index])))
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
    def _backward(self, activation, label, learning_rate, weight_decay, N=None):
        # forward pass
        activation, weighted_inputs, activations = self._forward(activation, include=True)
        # compute cost of forward pass for verbose output

        reg_term = 0
        if N is not None: # if regularization
            for weights_index, weights in enumerate(self.weights):
                if weights_index > 1:
                    reg_term += ((weight_decay / (2*N)) * np.dot(weights, weights.transpose()).sum())
        cost = self.loss.cost(activation, label) + reg_term
        # backward pass
        # final layer
        delta = np.multiply(self.loss.loss_prime(activation, label), self.activation_funcs[-1].function_prime(weighted_inputs[-1]))
        #remaining layers
        for layer_index in range(self.num_layers, 1, -1):
            # compute product before weights change
            product = np.dot(self.weights[layer_index].transpose(), delta)
            m = activations[layer_index-1].shape[1] # batch_size
            weight_gradient = (np.dot(delta, activations[layer_index-1].transpose()))*(1/m)
            bias_gradient = (delta).mean(axis=1, keepdims=True)
            self.weights[layer_index] = (weight_decay)*self.weights[layer_index] - (learning_rate*weight_gradient)
            self.biases[layer_index] -= (learning_rate)*bias_gradient
            # computes (layer_index - 1) delta vector
            if layer_index != 2: delta = np.multiply(product, self.activation_funcs[layer_index-1].function_prime(weighted_inputs[layer_index-1]))
        return cost

    def train(self, train_data, train_labels, valid_data, valid_labels, batch_size, learning_rate, weight_decay, epochs, verbose=False, plot_learning=False):
        train_cost_history, valid_cost_history = [], []
        N = train_data.shape[1]
        for epoch in range(epochs):
            # TODO stochastic gradient descent (how to keep labels with right data)
            for batch_index in range(train_data.shape[1]//batch_size):
                train_data_batch = train_data[:, range(batch_index*batch_size, ((batch_index+1)*batch_size))]
                labels_batch = train_labels[:, range(batch_index*batch_size, ((batch_index+1)*batch_size))]
                train_cost = self._backward(train_data_batch, labels_batch, learning_rate, weight_decay=weight_decay, N=N)
            train_cost_history.append(train_cost)
            if valid_data is not None:
                # validation performance
                validation_inferences = self._forward(activation=valid_data)
                reg_term = 0
                for weights_index, weights in enumerate(self.weights):
                    if weights_index > 1: reg_term += ((weight_decay / (2*N)) * np.dot(weights, weights.transpose()).sum())
                validation_cost = self.loss.cost(validation_inferences, valid_labels) + reg_term
                valid_cost_history.append(validation_cost)
            if verbose and (epoch % 5) == 0: 
                print(f'Training cost after epoch {epoch} = {train_cost}') 
                if valid_data is not None: print(f'Validation cost after epoch {epoch} = {validation_cost}') 
        
        if plot_learning: # plot learning curves
            plt.plot([i for i in range(1, epochs+1)], train_cost_history, label=f'Train')
            if valid_data is not None:  plt.plot([i for i in range(1, epochs+1)], valid_cost_history, label=f'Validation')
            plt.title(f'Training and validation cost per epoch')
            plt.legend(loc='upper right')
            plt.xlabel(f'Epoch')
            plt.ylabel(f'Cost (MSE)')
            plt.show()

    def inference(self, data):
        return np.argmax(self._forward(data), axis=0, keepdims=True)

    def test(self, test_data, test_labels, verbose=False):
        results_vector = self.inference(data=test_data)
        correct_inferences = np.equal(results_vector, test_labels).sum()
        total_inferences = (test_labels.shape[1])
        if verbose: print(f'Correct inferences={correct_inferences} out of {total_inferences} total inferences.')
        return correct_inferences/total_inferences

    def num_params(self):
        num_parameters = 0
        for weight_matrix in self.weights[2:]:
            dims = weight_matrix.shape
            num_parameters += (dims[0] * (dims[1]+1)) # +1 for biases 
        return num_parameters

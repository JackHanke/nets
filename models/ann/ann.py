# this code was created following the lecture notes found here: https://sgfin.github.io/files/notes/CS321_Grosse_Lecture_Notes.pdf
import numpy as np
import random
from math import sqrt
import json

class ArtificialNeuralNetwork:
    # dims is tuple of length >=2 that defines the model dimensions
    #   ie. (784, 15, 10) means a 784 x 15 array and a 15 x 10 array 
    # activations is tuple of activation function objects
    # loss is a tuple of a loss function and its derivative that accepts an activation vector and label vector
    def __init__(self, dims, activation_funcs, loss, seed=None, version_num=0, load_loc=None):
        if load_loc is None: 
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
        else: 
            with open(load_loc, 'r') as fin:
                model_info_dict = json.load(fin)
                self.weights

        def save(self, loc):
            if loc is None: save_loc_part = f'models/ann/ann-0'
            else: save_loc_part = loc

            with open(save_loc_part+'.json', 'w') as fout:
                model_info_dict = {
                    'weights':self.weights, 
                    'biases':self.biases, 
                    'activation_functs':self.activation_funcs,
                    '':1
                    }
                json.dump(model_info_dict, fout)

            with open(save_loc_part+'-params.json', 'w') as fout:
                params_dict = {
                    "version_num": self.version_num,
                    "lmbda": self.lmbda,
                    "n_step": self.n_step,
                    "discounting_param": self.discounting_param,
                    "reward_scale": self.reward_scale,
                    "learning_rate": self.learning_rate,
                    "state_value_function_approx": self.state_value_function_approx.name
                }
                json.dump(params_dict, fout)

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
            # if layer_index != 2: delta = np.multiply(product, self.activation_funcs[layer_index-1][1](weighted_inputs[layer_index-1]))
            if layer_index != 2: delta = np.multiply(product, self.activation_funcs[layer_index-1].function_prime(weighted_inputs[layer_index-1]))
            # print(f'norm of weight gradient at layer {layer_index} = {np.linalg.norm(weight_gradient)}')
        return cost

    def train(self, train_data, labels, batch_size, learning_rate, weight_decay, epochs, verbose=False):
        for epoch in range(epochs):
            # TODO stochastic gradient descent (how to keep labels with right data)
            for batch_index in range(train_data.shape[1]//batch_size):
                train_data_batch = train_data[:, range(batch_index*batch_size, ((batch_index+1)*batch_size))]
                labels_batch = labels[:, range(batch_index*batch_size, ((batch_index+1)*batch_size))]
                cost = self._backward(train_data_batch, labels_batch, learning_rate, weight_decay=weight_decay, N=train_data.shape[1])
            if verbose and (epoch % 10) == 0: print(f'Cost after epoch {epoch} = {cost}') 

    def inference(self, data):
        return np.argmax(self._forward(data), axis=0, keepdims=True)

    def test(self, test_data, test_labels, verbose=False):
        results_vector = self.inference(data=test_data)
        correct_inferences = np.equal(results_vector, test_labels).sum()
        total_inferences = (test_labels.shape[1])
        if verbose: print(f'Correct inferences={correct_inferences} out of {total_inferences} total inferences.')
        return correct_inferences/total_inferences

if __name__ == '__main__':
    mnist = False
    iris = True

    if mnist:
        # test network on MNIST dataset
        from dataload import read_images_labels
        k = 10 # k-hot value
        # load train
        training_images_filepath, training_labels_filepath = './data/train-images-idx3-ubyte/train-images-idx3-ubyte', './data/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
        # training_images_filepath, training_labels_filepath = '~/vault/software/mnist/data/train-images-idx3-ubyte/train-images-idx3-ubyte', '~/vault/software/mnist/data/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
        x_train, y_train = read_images_labels(training_images_filepath, training_labels_filepath)

        # load test
        test_images_filepath, test_labels_filepath = './data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte', './data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
        # test_images_filepath, test_labels_filepath = '~/vault/software/mnist/data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte', '~/vault/software/mnist/data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
        x_test, y_test = read_images_labels(test_images_filepath, test_labels_filepath)

        # reformat data for model
        x_train = x_train.transpose() * (1/255)

        # number of training examples N
        N = x_train.shape[1]

        # reformat data to k-hot format TODO: does numpy have a better way to do this?
        temp_array = np.zeros((k, N))
        for index,val in enumerate(y_train):
            temp_array[val][index] = 1
        y_train = temp_array
        x_test, y_test = (x_test.transpose() * (1/255)), y_test.reshape(1,-1)
        print('MNIST data loaded in.')

        learning_rate = 0.5
        # train on data with following parameters
        epochs = 90
        batch_size = 10
        ## initialize network
        network = Network(
            dims=(784,30,10), \
            activation_funcs = [(sigmoid, sigmoid_prime),(sigmoid, sigmoid_prime)], 
            loss=(cross_entropy_loss, cross_entropy_loss_prime), 
            cost=cost, 
            weight_decay=(1-(5*learning_rate)/N),
            seed=1 
        )

        np.set_printoptions(suppress=True, linewidth = 150)

        print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate={learning_rate}')
        network.train(
            train_data=x_train, 
            labels=y_train, 
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            epochs=epochs, 
            verbose=True
        )
        print('Training completed.')

        # test performance
        accuracy = network.test(test_data=x_test, test_labels=y_test, verbose=True)
        print(f'Training resulted in network with {accuracy*100 :.4}% accuracy.')

    

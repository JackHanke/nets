import numpy as np
from time import time
import matplotlib.pyplot as plt
import pickle
from ..ann.ann import ArtificialNeuralNetwork

def train_cnn(model, train_data, train_labels, valid_data, valid_labels, batch_size, epochs, optimizer, verbose=False, plot_learning=False, N=None):
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

class ConvolutionalNeuralNetwork(ArtificialNeuralNetwork):
    def __init__(self):
        pass

    def _forward(self):
        pass

    def _backward(self):
        pass

    def num_params(self):
        pass

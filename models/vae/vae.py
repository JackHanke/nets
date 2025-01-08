import numpy as np
from time import time
import matplotlib.pyplot as plt

class VariationalAutoEncoder:
    def __init__(self, encodernet, decodernet, version_num=0):
        # dimension of latent space
        self.latent_dim = decodernet.dims[0]
        self.encodernet = encodernet
        self.decodernet = decodernet
        self.version_num = version_num

    # make sample z from parameter information
    def _params_vec_to_sample(self, params_vec, noise=None):
        mu = params_vec[:self.latent_dim]
        logsig = params_vec[self.latent_dim:]
        # get noise
        if noise is None: epsilon = np.random.normal(loc=0, scale=1, size=(logsig.shape))
        elif noise is not None: epsilon = noise
        # latent variable calculation from noise
        z = mu + np.multiply(np.exp(logsig), epsilon)
        return z, epsilon

    #   
    def _forward(self, activation, include=False, noise=None):
        # run encoder to generate mu and log(sigma)
        # params_vec, encoder_winputs, encoder_activations = self.encodernet._forward(activation, include=include)
        params_vec = self.encodernet._forward(activation, include=False)
        # sample
        z, epsilon = self._params_vec_to_sample(params_vec=params_vec, noise=noise)
        # run decoder to predict image
        # activation, decoder_winputs, decoder_activations = self.decodernet._forward(z, include=include)
        activation = self.decodernet._forward(z, include=False)
        
        return activation

    # TODO rewrite for VAE
    # forward and backward pass
    def _backward(self, activation, label, learning_rate, weight_decay, N=None):
        # forward pass of encoder
        params_vec = self.encodernet._forward(activation=activation) # TODO this runs twice, fix!
        # get sample vector z
        z, epsilon = self._params_vec_to_sample(params_vec=params_vec)

        # backprop the decoder
        # NOTE I use the presense of the epsilon argument to 
        rec_cost, z_delta = self.decodernet._backward(
            activation=z, # z
            label=label, # 
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # backprop the encoder
        reg_cost, _ = self.encodernet._backward(
            activation=activation,
            label=z_delta,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epsilon=epsilon
        )

        # compute cost of forward pass for verbose output
        reg_term = 0
        if N is not None: # if regularization
            for weights_index, weights in enumerate(self.encodernet.weights):
                if weights_index > 1:
                    reg_term += ((weight_decay / (2*N)) * np.dot(weights, weights.transpose()).sum())
            for weights_index, weights in enumerate(self.decoder.weights):
                if weights_index > 1:
                    reg_term += ((weight_decay / (2*N)) * np.dot(weights, weights.transpose()).sum())

        cost = rec_cost + reg_term
        return cost

    # TODO dont repeat yourself
    def train(self, train_data, train_labels, valid_data, valid_labels, batch_size, learning_rate, weight_decay, epochs, verbose=False, plot_learning=False, N=None):
        train_cost_history, valid_cost_history = [], []
        if N is not None: N = train_data.shape[1]
        for epoch in range(epochs):
            start = time()
            # TODO make this stochastic 
            for batch_index in range(train_data.shape[1]//batch_size):
                train_data_batch = train_data[:, range(batch_index*batch_size, ((batch_index+1)*batch_size))]
                labels_batch = train_labels[:, range(batch_index*batch_size, ((batch_index+1)*batch_size))]
                train_cost = self._backward(
                    activation=train_data_batch,
                    label=labels_batch,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    N=N
                )
            train_cost_history.append(train_cost)
            end = time()
            if valid_data is not None:
                # validation performance
                validation_inferences = self._forward(activation=valid_data)
                reg_term = 0
                # TODO fix reg
                # for weights_index, weights in enumerate(self.weights):
                #     if weights_index > 1: reg_term += ((weight_decay / (2*N)) * np.dot(weights, weights.transpose()).sum())
                validation_cost = self.decodernet.loss.cost(validation_inferences, valid_labels) + reg_term
                valid_cost_history.append(validation_cost)
            if verbose and (epoch % 3) == 0: 
                print(f'Training cost after epoch {epoch} = {train_cost:.6f}. Completed in {end-start:.4f}s') 
                if valid_data is not None: print(f'Validation cost after epoch {epoch} = {validation_cost:.6f}') 
        
            if plot_learning and (epoch % 3) == 0: # plot learning curves
                plt.plot([i for i in range(1, len(train_cost_history)+1)], train_cost_history, label=f'Train')
                if valid_data is not None:  plt.plot([i for i in range(1, len(valid_cost_history)+1)], valid_cost_history, label=f'Validation')
                plt.title(f'Epoch {epoch} has train cost {train_cost:.6f}')
                plt.legend(loc='upper right')
                plt.xlabel(f'Epoch')
                plt.ylabel(f'Cost (MSE)')
                plt.pause(0.000001)
            if plot_learning and epoch != epochs-1: plt.cla()
        if plot_learning:
            plt.title(f'Completed {epochs} epochs at train cost {train_cost:.6f}.')
            plt.show()

    def encode(self, activation, noise=None):
        params_vec = self.encodernet._forward(activation, include=False)
        # sample
        z, epsilon = self._params_vec_to_sample(params_vec=params_vec, noise=noise)
        return z

    def decode(self, activation):
        # activation is the typically denoted 'z'
        activation = self.decodernet._forward(activation, include=False)
        return activation

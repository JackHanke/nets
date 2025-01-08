from ..ann.ann import ArtificialNeuralNetwork
import numpy as np

class AutoEncoder(ArtificialNeuralNetwork):
    def __init__(self, add_noise=False, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        # NOTE ! this class assumes the ann dims arg has odd length
        self.add_noise = add_noise
        self.cutoff = ((self.num_layers)//2) + 2

    def _forward(self, activation, include=False):
        weighted_inputs = [-1, activation]
        activations = [-1, activation]
        for layer_index in range(2,self.num_layers+1):
            weighted_input = np.dot(self.weights[layer_index], activation) + \
                            np.dot(self.biases[layer_index], np.ones((1,activation.shape[1])))
            weighted_inputs.append(weighted_input)
            activation = self.activation_funcs[layer_index].function(weighted_input)
            if self.add_noise and layer_index == self.cutoff:
                activation += np.random.normal(loc=(1/2), scale=(1/6), size=activation.shape)
            activations.append(activation)
        if include: return activation, weighted_inputs, activations
        else: return activation

    def encode(self, activation):
        for layer_index in range(2, self.cutoff):
            weighted_input = np.dot(self.weights[layer_index], activation) + \
                            np.dot(self.biases[layer_index], np.ones((1,activation.shape[1])))
            activation = self.activation_funcs[layer_index].function(weighted_input)
        return activation

    def decode(self, activation):
        for layer_index in range(self.cutoff, self.num_layers+1):
            weighted_input = np.dot(self.weights[layer_index], activation) + \
                            np.dot(self.biases[layer_index], np.ones((1,activation.shape[1])))
            activation = self.activation_funcs[layer_index].function(weighted_input)
        return activation

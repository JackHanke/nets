from ..ann.ann import ArtificialNeuralNetwork
import numpy as np

class AutoEncoder(ArtificialNeuralNetwork):
    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        # NOTE ! this class assumes the ann dims arg has odd length
        self.cutoff = ((self.num_layers)//2) + 2

    def encoder_inference(self, activation):
        for layer_index in range(2, self.cutoff):
            weighted_input = np.dot(self.weights[layer_index], activation) + \
                            np.dot(self.biases[layer_index], np.ones((1,activation.shape[1])))
            activation = self.activation_funcs[layer_index].function(weighted_input)
        return activation

    def decoder_inference(self, activation):
        for layer_index in range(self.cutoff, self.num_layers+1):
            weighted_input = np.dot(self.weights[layer_index], activation) + \
                            np.dot(self.biases[layer_index], np.ones((1,activation.shape[1])))
            activation = self.activation_funcs[layer_index].function(weighted_input)
        return activation

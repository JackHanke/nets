from ..ann.ann import ArtificialNeuralNetwork
import numpy as np

epsilon = np.randon.normal(loc=0, scale=1, size=(1))

class VariationalAutoEncoder(ArtificialNeuralNetwork):
    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)

    def encoder_inference(self, activation):
        pass

    def decoder_inference(self, activation):
        pass


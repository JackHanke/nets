from ..ann.ann import ArtificialNeuralNetwork
import numpy as np



class VariationalAutoEncoder:
    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)

        self.encodernet = ArtificialNeuralNetwork()
        self.decodernet = ArtificialNeuralNetwork()

        self.loss = 0

    def encoder_inference(self, activation):
        params_vec = self.encoder_inference.inference(activation)
        mu = params_vec[:]
        logsig = params_vec[:]
        epsilon = np.randon.normal(loc=0, scale=1, size=(1))

        sample = g(epsilon)

        inf = self.encoder_inference.inference(sample)



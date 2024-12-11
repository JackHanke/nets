from ..ann.ann import ArtificialNeuralNetwork
from functions.activation_funcs import Sigmoid
from functions.loss_funcs import MSE
import numpy as np

class VariationalAutoEncoder:
    def __init__(self):
        # dimension of latent space
        self.latent_dim = 5
        self.encodernet = ArtificialNeuralNetwork(
            dims=(784, 32, 2*self.latent_dim),
            activation_funcs = [Sigmoid(), Sigmoid()], 
            loss=(MSE()), 
            seed=1,
            version_num=0
        )
        self.decodernet = ArtificialNeuralNetwork(
            dims=(self.latent_dim, 32, 784),
            activation_funcs = [Sigmoid(), Sigmoid()], 
            loss=(MSE()), 
            seed=1,
            version_num=0
        )

    def forward(self, activation):
        # run encoder net to generate mu and log(sigma)
        params_vec = self.encodernet._forward(activation)
        # split parameters
        mu = params_vec[:self.latent_dim]
        logsig = params_vec[self.latent_dim:]
        # get noise
        epsilon = np.random.normal(loc=0, scale=1, size=(self.latent_dim, 1))
        # latent variable calculation from noise
        z = mu + np.multiply(np.exp(logsig), epsilon)
        #
        prob_epsilon = 0
        # 
        inf = self.decodernet._forward(z)


        self.encodernet._backward()



        return inf





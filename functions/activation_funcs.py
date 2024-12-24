import numpy as np
# define activation functions

class Identity:
    def __init__(self):
        self.name = 'identity'
    
    def function(self, x):
        return x

    def function_prime(self, x):
        return np.ones(x.shape)

class Sigmoid:
    def __init__(self):
        self.name = 'sigmoid'
    
    def function(self, x):
        return 1/(1+np.exp(-x))

    def function_prime(self, x):
        sig = self.function(x)
        return sig*(1-sig)

class TanH:
    def __init__(self):
        self.name = 'hyperbolic tangent'
    
    def function(self, x):
        return np.tanh(x)

    def function_prime(self, x):
        tanh = self.function(x)
        return 1-np.square(tanh)

class ReLu:
    def __init__(self):
        self.name = 'relu'
    
    def function(self, x):
        return x * (x > 0)

    def function_prime(self, x):
        return 1. * (x > 0)


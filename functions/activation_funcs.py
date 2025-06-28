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
    def __init__(self, scale=1):
        self.name = 'hyperbolic tangent'
        self.scale = scale # TODO question your entire career path
    
    def function(self, x):
        return self.scale*np.tanh(x)

    def function_prime(self, x):
        temp = self.function(x)/self.scale
        return self.scale*(1-np.square(temp))

class ReLu:
    def __init__(self):
        self.name = 'relu'
    
    def function(self, x):
        return x * (x > 0)

    def function_prime(self, x):
        return 1. * (x > 0)

class LeakyReLu:
    def __init__(self):
        self.name = 'leaky relu'
    
    def function(self, x):
        return x * (x > 0) + 0.05*x*(x <= 0)

    def function_prime(self, x):
        return 1. * (x > 0) + 0.05 * (x <= 0)

class Swish:
    def __init__(self):
        self.name = 'swish'
    
    def function(self, x):
        val = x*(1/(1+np.exp(-x)))
        # TODO remove
        if np.isnan(np.sum(val)): input(x)
        # print((np.max(x), np.min(x)))
        return val

    def function_prime(self, x):
        return 0.5 + (x+np.sinh(x) /(4*np.square(np.cosh(0.5*x))))

class SoftMax:
    def __init__(self):
        self.name = 'softmax'

    # expects x to be of dimension (batch, row, axis to be summed over)
    def function(self, x: np.array):
        # numerical stability
        m = np.max(x, axis=2, keepdims=True)

        numerator = np.exp(x - m)
        denominator = np.sum(numerator, axis=2, keepdims=True)
        # numerator
        val = numerator / denominator
        return val

    def function_prime(self, x: np.array):
        # NOTE I'm not 100% sure of this
        val = self.function(x) * (1 - self.function(x))
        return val



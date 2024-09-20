import numpy as np
# define loss & cost function and derivatives

class MSE:
    def __init__(self):
        self.name = 'mse'

    def loss(self, activation, label):
        return 0.5*(activation-label)**2

    def loss_prime(self, activation, label):
        return (activation-label)

    def cost(self, activation, label):
        return np.average(self.loss(activation, label))

# TODO figure out why this doesnt work
class CrossEntropy:
    def __init__(self):
        self.name='Cross Entropy'

    def loss(self, activation, label):
        return -1*(label*np.log(activation) + \
            (np.ones(label.shape)-label)*np.log(np.ones(activation.shape)-activation))
    
    def loss_prime(self, activation, label):
        return np.average(self.loss(activation, label))

    def cost(self, activation, label):
        return np.average(self.loss(activation, label))

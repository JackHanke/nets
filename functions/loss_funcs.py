import numpy as np
# define loss & cost function and derivatives

class MSE:
    def __init__(self):
        self.name = 'mse'

    def loss(self, activation, label):
        return 0.5*(activation-label)**2

    def loss_prime(self, activation, label, **kwargs):
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
    
    def loss_prime(self, activation, label, **kwargs):
        return np.average(self.loss(activation, label))

    def cost(self, activation, label):
        return np.average(self.loss(activation, label))

class VAEInternal:
    def __init__(self, latent_dim, reg_weight_update):
        self.name = 'VAE internal loss'
        self.latent_dim = latent_dim
        self.reg_weight = 0
        self.reg_weight_update = reg_weight_update

    def loss(self, activation, label=None):
        mu = activation[:self.latent_dim]
        logsig = activation[self.latent_dim:]
        sig = np.exp(logsig)
        return -0.5*(np.ones(mu.shape) + 2*logsig - np.square(mu) - np.square(sig))

    def loss_prime(self, activation, label, epsilon=None, **kwargs):
        # here the label is dC/dz, where z = mu + sig*epsilon
        mu = activation[:self.latent_dim]
        logsig = activation[self.latent_dim:]
        sig = np.exp(logsig)

        # Reconstruction term, NOTE a little ugly
        temp = np.multiply(epsilon, sig)
        loss_prime_rec_term = np.vstack((label, np.multiply(label, temp)))

        # Regularization term
        loss_prime_reg_term = np.vstack((mu, np.exp(2*logsig) - np.ones(logsig.shape)))

        # if self.reg_weight < 1: self.reg_weight += self.reg_weight_update
        # print(f' > Regularization Weight {self.reg_weight}')

        # weight and add the two terms
        # return loss_prime_rec_term + self.reg_weight*loss_prime_reg_term
        return loss_prime_rec_term + loss_prime_reg_term

    def cost(self, activation, label):
        return np.sum(self.loss(activation, label))
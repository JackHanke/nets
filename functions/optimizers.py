import numpy as np

# optimizer classes
# TODO make all of this parallelized, collect all parameters as one vector? is that faster? good start: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html

# Stochastic Gradient Descent Optimizer class
class SGD:
    def __init__(self, learning_rate, weight_decay):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def step(self, weights, weights_gradients, biases, biases_gradients):
        for layer_index in range(len(weights)-1, 1, -1):
            weights[layer_index] = ((self.weight_decay)*weights[layer_index]) - (self.learning_rate*weights_gradients[layer_index])
            biases[layer_index] = biases[layer_index] - (self.learning_rate)*biases_gradients[layer_index]
        return weights, biases

# ADAM Optimizer class
class ADAM:
    def __init__(self, weights, biases, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.m = [-1 if type(weight) is int else np.zeros(weight.shape) for weight in weights], [-1 if type(bias) is int else np.zeros(bias.shape) for bias in biases]
        self.v = [-1 if type(weight) is int else np.zeros(weight.shape) for weight in weights], [-1 if type(bias) is int else np.zeros(bias.shape) for bias in biases]

    def step(self, weights, weights_gradients, biases, biases_gradients):
        self.t += 1
        for layer_index in range(len(weights)-1, 1, -1):
            self.m[0][layer_index] = self.beta_1 * self.m[0][layer_index] + (1-self.beta_1) * weights_gradients[layer_index]
            self.m[1][layer_index] = self.beta_1 * self.m[1][layer_index] + (1-self.beta_1) * biases_gradients[layer_index]
            self.v[0][layer_index] = self.beta_2 * self.v[0][layer_index] + (1-self.beta_2) * np.square(weights_gradients[layer_index])
            self.v[1][layer_index] = self.beta_2 * self.v[1][layer_index] + (1-self.beta_2) * np.square(biases_gradients[layer_index])
            weights[layer_index] -= self.alpha * self.m[0][layer_index] / (1 - self.beta_1**self.t) / (np.sqrt(self.v[0][layer_index]/(1-self.beta_2**self.t)) + self.epsilon)
            biases[layer_index] -= self.alpha * self.m[1][layer_index] / (1 - self.beta_1**self.t) / (np.sqrt(self.v[1][layer_index]/(1-self.beta_2**self.t)) + self.epsilon)
        return weights, biases

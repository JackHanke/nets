# optimizer classes

class SGD:
    def __init__(self, learning_rate, weight_decay):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def step(self, weights, weights_gradients, biases, biases_gradients):
        for layer_index in range(len(weights)-1, 1, -1):
            weights[layer_index] = ((self.weight_decay)*weights[layer_index]) - (self.learning_rate*weights_gradients[layer_index])
            biases[layer_index] = biases[layer_index] - (self.learning_rate)*biases_gradients[layer_index]
        return weights, biases

class ADAM:
    def __init__(self):
        pass

    def step(self):
        pass

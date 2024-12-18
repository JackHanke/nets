from ..ann.ann import ArtificialNeuralNetwork
import numpy as np

# Constructed from: https://arxiv.org/pdf/2208.11970

class Diffusion(ArtificialNeuralNetwork):
    def __init__(self, T, **kwargs):
        super(Diffusion, self).__init__(**kwargs)
        self.T = T # NOTE the number of diffusion steps

    def gen(self, condition=None, noise=None, plotgen=False):
        # NOTE also horrible
        if noise is None: x_vec = noise
        elif noise is None: x_vec = np.random.normal(loc=0, scale=1, size=(784,1))
        condition_vec = np.zeros((10,1))
        if condition is not None: condition_vec[condition] = 1
        x_vec = np.vstack((x_vec, condition_vec))
        print(x_vec.shape)
        for t in range(self.T): 
            x_vec = self._forward(activation=x_vec)
            print(x_vec.shape)
        return x_vec

# NOTE this is horrible
def prep_data_for_diffusion(x, y, T):
    x = x.transpose()
    y = y.transpose()

    return_vec = []
    for data_vec, label_vec in zip(x, y):
        noise = np.random.normal(loc=0, scale=1, size=data_vec.shape)
        for t in range(T+1):
            temp_vec = ((T-t)/T)*data_vec + (t/T)*noise
            temp_vec = np.reshape(temp_vec, (784, 1)) # TODO this is worse
            label_vec = np.reshape(label_vec, (10, 1))
            temp_vec = np.vstack((temp_vec, label_vec))
            return_vec.append(temp_vec)

    return_vec = np.array(return_vec).transpose()
    return_vec = np.reshape(return_vec, (794, -1))
    return return_vec


if __name__ == '__main__':
    x_vec = np.random.normal(loc=0, scale=1, size=(784,1))
    condition_vec = np.zeros((10,1))
    x_vec = np.vstack((x_vec, condition_vec))
    print(x_vec.shape)

from models.ann.ann import ArtificialNeuralNetwork
from models.diffusion.diffusion import Diffusion

from functions.activation_funcs import *
from functions.loss_funcs import *
import numpy as np
from time import time

def mnist_benchmark(network, save=False):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )
    print('MNIST data loaded in.')

if __name__ == '__main__':
    pass

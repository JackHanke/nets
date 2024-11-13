from models.ann.ann import ArtificialNeuralNetwork
from functions.activation_funcs import Sigmoid
from functions.loss_funcs import MSE
import numpy as np
from time import time

from datasets.mnist.dataload import get_mnist_data

def mnist_benchmark(network, save=False):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )
    print('MNIST data loaded in.')

    # train on data with following parameters
    learning_rate = 0.05
    epochs = 25
    batch_size = 10

    print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate={learning_rate}')
    start = time()
    network.train(
        train_data=x_train, 
        train_labels=y_train,
        valid_data=x_valid,
        valid_labels=y_valid,
        batch_size=batch_size, 
        learning_rate=learning_rate, 
        weight_decay=(1-(5*learning_rate)/(x_train.shape[1])),
        epochs=epochs, 
        verbose=True,
        plot_learning=False
    )
    print(f'Training completed in {((time()-start)/60):.4f} minutes.')

    # test performance
    accuracy = network.test(test_data=x_test, test_labels=y_test, verbose=True)
    print(f'Training resulted in network with {accuracy*100 :.4}% accuracy.')

    if save: network.save(loc=f'models/ann/mnist-ann')
    return accuracy

if __name__ == '__main__':
    network = ArtificialNeuralNetwork(
        dims=(784, 30, 10),
        activation_funcs = [Sigmoid(), Sigmoid()], 
        loss=(MSE()), 
        seed=1,
        version_num=0
    )
    acc = mnist_benchmark(network=network, save=False)

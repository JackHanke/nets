from models.ann.ann import ArtificialNeuralNetwork
from functions.activation_funcs import *
from functions.loss_funcs import *
import numpy as np
from time import time

def mnist_benchmark(network, save=False):
    from datasets.mnist.dataload import read_images_labels
    k = 10 # k-hot value
    # load train
    training_images_filepath, training_labels_filepath = './datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte', './datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    # training_images_filepath, training_labels_filepath = '~/vault/software/mnist/data/train-images-idx3-ubyte/train-images-idx3-ubyte', '~/vault/software/mnist/data/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    x_train, y_train = read_images_labels(training_images_filepath, training_labels_filepath)

    # load test
    test_images_filepath, test_labels_filepath = './datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte', './datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    # test_images_filepath, test_labels_filepath = '~/vault/software/mnist/data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte', '~/vault/software/mnist/data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    x_test, y_test = read_images_labels(test_images_filepath, test_labels_filepath)

    # reformat data for model
    x_train = x_train.transpose() * (1/255)

    # number of training examples N
    N = x_train.shape[1]

    # reformat data to k-hot format TODO: does numpy have a better way to do this?
    temp_array = np.zeros((k, N))
    for index,val in enumerate(y_train):
        temp_array[val][index] = 1
    y_train = temp_array
    x_test, y_test = (x_test.transpose() * (1/255)), y_test.reshape(1,-1)
    print('MNIST data loaded in.')

    # train on data with following parameters
    learning_rate = 0.05
    epochs = 90
    batch_size = 10

    np.set_printoptions(suppress=True, linewidth = 150)

    print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate={learning_rate}')
    network.train(
        train_data=x_train, 
        labels=y_train, 
        batch_size=batch_size, 
        learning_rate=learning_rate, 
        weight_decay=(1-(5*learning_rate)/N),
        epochs=epochs, 
        verbose=True
    )
    print('Training completed.')

    # test performance
    accuracy = network.test(test_data=x_test, test_labels=y_test, verbose=True)
    print(f'Training resulted in network with {accuracy*100 :.4}% accuracy.')

    if save: network.save(loc=f'models/ann/mnist-ann')
    return accuracy

if __name__ == '__main__':
    network = ArtificialNeuralNetwork(
        dims=(784,30,10),
        activation_funcs = [Sigmoid(),Sigmoid()], 
        loss=(MSE()), 
        seed=1,
        version_num=0
    )
    mnist_benchmark(network=network, save=False)

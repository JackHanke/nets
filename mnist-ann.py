from models.ann.ann import ArtificialNeuralNetwork, train_ann, test_ann
from functions.activation_funcs import Sigmoid
from functions.loss_funcs import MSE
import numpy as np
from time import time
from datasets.mnist.dataload import get_mnist_data
from functions.optimizers import *
import pickle

def mnist_benchmark(path=None):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )
    print('MNIST data loaded in.')

    if path is None: # if no network path is given
        # make new network 
        network = ArtificialNeuralNetwork(
            dims = (784, 30, 10),
            activation_funcs = [Sigmoid(), Sigmoid()], 
            loss = MSE(), 
            seed = 1,
            version_num = 0
        )

        # set the optimizer

        # optimizer = SGD(
        #     learning_rate = 0.1,
        #     weight_decay = 0.99999
        # ) 
        # print(f'Optimizer ')

        optimizer = ADAM(
            weights=network.weights,
            biases=network.biases
        )

        # train on data with following parameters
        epochs = 100
        batch_size = 128

        print(f'Beginning training for {epochs} epochs at batch size {batch_size}')
        start = time()
        train_ann(
            model = network,
            train_data = x_train, 
            train_labels = y_train,
            valid_data = x_valid,
            valid_labels = y_valid,
            batch_size = batch_size, 
            epochs = epochs,
            optimizer = optimizer,
            verbose = True,
            plot_learning = False,
            N=x_train.shape[1]
        )
        print(f'Training completed in {((time()-start)/60):.4f} minutes.')
        
        path_str = f'models/ann/saves/mnist_ann_{network.version_num}.pkl'
        with open(path_str, 'wb') as f:
            pickle.dump(network, file=f)
        print(f'Model saved at: {path_str}')

    elif path:
        # load in already trained model
        with open(path, 'rb') as f:
            network = pickle.load(f)

    # test performance
    accuracy = test_ann(
        model = network,
        test_data = x_test,
        test_labels = y_test,
        verbose = True
    )
    print(f'Training resulted in network with {accuracy*100 :.4}% accuracy.')
    return accuracy

if __name__ == '__main__':
    acc = mnist_benchmark(path=None)

    # path = f'models/ann/saves/mnist_ann_0.pkl'
    # acc = mnist_benchmark(path=path)

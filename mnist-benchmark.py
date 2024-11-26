from models.ann.ann import ArtificialNeuralNetwork
from functions.activation_funcs import Sigmoid
from functions.loss_funcs import MSE
import numpy as np
from time import time
from datasets.mnist.dataload import get_mnist_data
import pickle

def mnist_benchmark(network_path=None):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )
    print('MNIST data loaded in.')

    if network_path is None: # if no network path is given
        # make new network 
        network = ArtificialNeuralNetwork(
            dims=(784, 30, 10),
            activation_funcs = [Sigmoid(), Sigmoid()], 
            loss=(MSE()), 
            seed=1,
            version_num=0
        )

        # train on data with following parameters
        learning_rate = 0.05
        epochs = 25
        batch_size = 10

        print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate = {learning_rate}')
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
        
        path_str = f'models/ann/saves/mnist_ann_{network.version_num}.pkl'
        with open(path_str, 'wb') as f:
            pickle.dump(network, file=f)
        print(f'Model saved at: {path_str}')

    elif network_path:
        # load in already trained model
        with open(network_path, 'rb') as f:
            network = pickle.load(f)

    # test performance
    accuracy = network.test(test_data=x_test, test_labels=y_test, verbose=True)
    print(f'Training resulted in network with {accuracy*100 :.4}% accuracy.')
    return accuracy

if __name__ == '__main__':
    # acc = mnist_benchmark(network_path=None, save=True)

    network_path = f'models/ann/saves/mnist_ann_0.pkl'
    acc = mnist_benchmark(network_path=network_path, save=False)

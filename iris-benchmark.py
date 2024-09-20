from models.ann.ann import ArtificialNeuralNetwork
from functions.activation_funcs import *
from functions.loss_funcs import *
import numpy as np
from time import time

def iris_benchmark(network, save=False):
    # iris dataset
    k = 3 # k-hot value

    # csv columns are sepal_length,sepal_width,petal_length,petal_width,species
    iris_dataset = np.genfromtxt('datasets/iris/iris.csv', delimiter=',')

    # randomize and split
    np.random.seed(1)
    np.random.shuffle(iris_dataset)

    data = iris_dataset[:, range(4)]
    labels = iris_dataset[:, [4]]

    data = data.transpose()
    labels = labels.transpose()

    # split test and train data at 60%
    total = iris_dataset.shape[0]
    split = total//5 * 3
    x_train, x_test = data[:, range(0, split)], data[:, range(split, total)]
    labels_train, y_test = labels[:, range(0, split)], labels[:, range(split, total)]

    labels_train = labels_train.transpose()
    y_train = np.zeros((k, split))
    for index, val in enumerate(labels_train): 
        y_train[int(val[0])][index] = 1

    # train on data with following parameters
    epochs = 300
    learning_rate = 0.1
    batch_size = 5

    np.set_printoptions(suppress=True, linewidth=150)

    print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate={learning_rate}')
    start = time()
    network.train(
        train_data=x_train, 
        labels=y_train, 
        batch_size=batch_size, 
        learning_rate=learning_rate, 
        weight_decay=1,
        epochs=epochs, 
        verbose=True
    )
    print(f'Training completed after {(time()-start):.4f} seconds.')

    # test performance
    accuracy = network.test(
        test_data=x_test, 
        test_labels=y_test, 
        verbose=True
    )
    print(f'Training resulted in network with {accuracy*100 :.4}% accuracy.')

    if save: network.save(loc=f'models/ann/iris-ann')
    return accuracy



if __name__ == '__main__':
    network =ArtificialNeuralNetwork(
        dims=(4,5,3),
        activation_funcs = [Sigmoid(),Sigmoid()],
        loss=(MSE()),
        seed=1,
        version_num=0
    )
    iris_benchmark(network=network, save=False)



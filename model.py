# this code was created following the lecture notes found here: https://sgfin.github.io/files/notes/CS321_Grosse_Lecture_Notes.pdf
import numpy as np
import random

# define activation functions
def sigmoid(x): return 1/(1+np.exp(-x))

def sigmoid_prime(x): 
    sig = sigmoid(x)
    return sig * (1-sig)

# define loss & cost function and derivative
def mse_loss(activation, label): return 0.5*(activation-label)**2
def mse_loss_prime(activation, label): return (activation-label)

def cost(loss, activation, label): return np.average(loss(activation, label))
# def cost(loss): return np.average(loss)

class Network:
    # dims is tuple of length >=2 that defines the model dimensions
    #   ie. (784, 15, 10) means a 784 x 15 array and a 15 x 10 array 
    # activations is tuple of tuples of vectorized activation functions and their derivatives
    # loss is a tuple of a loss function and its derivative that accepts an activation vector and label vector
    def __init__(self, dims, activation_funcs, loss, cost, seed=None):
        self.loss = loss[0]
        self.loss_prime = loss[1]
        self.cost = cost
        self.activation_funcs = [-1,-1] + activation_funcs # insert filler to align indexing with textbook
        if seed is not None: np.random.seed(seed)

        self.weights = [-1,-1] # insert filler to align indexing with textbook
        self.biases = [-1,-1]
        for dim_index in range(len(dims)-1):
            self.weights.append(np.random.normal(loc=0, scale=1, size=(dims[dim_index+1], dims[dim_index])))
            self.biases.append(np.random.normal(loc=0, scale=1, size=(dims[dim_index+1], 1)))

        self.num_layers = len(dims)

    # forward pass
    def _forward(self, activation, include=False):
        weighted_inputs = [-1, activation]
        activations = [-1, activation]
        for layer_index in range(2,self.num_layers+1):
            weighted_input = np.dot(self.weights[layer_index], activation) + \
                            np.dot(self.biases[layer_index], np.ones((1,activation.shape[1])))
            weighted_inputs.append(weighted_input)
            activation = self.activation_funcs[layer_index][0](weighted_input)
            activations.append(activation)
        if include: return activation, weighted_inputs, activations
        else: return activation

    # forward and backward pass
    def _backward(self, activation, label, learning_rate):
        # forward pass
        activation, weighted_inputs, activations = self._forward(activation, include=True)
        # compute cost of forward pass for verbose output
        cost = self.cost(self.loss, activation, label)
        # backward pass
        # final layer
        delta = np.multiply(self.loss_prime(activation, label), self.activation_funcs[-1][1](weighted_inputs[-1]))
        #remaining layers
        for layer_index in range(self.num_layers, 1, -1):
            # compute product before weights change
            product = np.dot(self.weights[layer_index].transpose(), delta)
            m = activations[layer_index-1].shape[1] # batch_size
            weight_gradient = (np.dot(delta, activations[layer_index-1].transpose()))
            bias_gradient = (delta).mean(axis=1, keepdims=True)
            # print('begin')
            # print(weight_gradient.shape)
            # print(bias_gradient.shape)
            # print(self.weights[layer_index].shape)
            # print(self.biases[layer_index].shape)
            # print('end')
            self.weights[layer_index] -= (learning_rate/m) * weight_gradient
            self.biases[layer_index] -= (learning_rate) * bias_gradient
            # computes (layer_index - 1) delta vector
            if layer_index != 2: delta = np.multiply(product, self.activation_funcs[layer_index-1][1](weighted_inputs[layer_index-1]))
            # print(f'norm of weight gradient at layer {layer_index} = {np.linalg.norm(weight_gradient)}')

        return cost

    def train(self, train_data, labels, batch_size, learning_rate, epochs, verbose=False):
        for epoch in range(epochs):
            # TODO stochastic gradient descent (how to keep labels with right data)
            for batch_index in range(train_data.shape[1]//batch_size):
                train_data_batch = train_data[:, range(batch_index*batch_size, ((batch_index+1)*batch_size))]
                labels_batch = labels[:, range(batch_index*batch_size, ((batch_index+1)*batch_size))]
                cost = self._backward(train_data_batch, labels_batch, learning_rate)
            if verbose and (epoch % 10) == 0: print(f'Cost after epoch {epoch} = {cost}') 

    def inference(self, data):
        return np.argmax(self._forward(data), axis=0, keepdims=True)

    def test(self, test_data, test_labels, verbose=False):
        results_vector = self.inference(data=test_data)
        correct_inferences = np.equal(results_vector, test_labels).sum()
        total_inferences = (test_labels.shape[1])
        if verbose: print(f'Correct inferences={correct_inferences} out of {total_inferences} total inferences.')
        return correct_inferences/total_inferences

if __name__ == '__main__':
    mnist = True
    iris = False

    if mnist:
        # test network on MNIST dataset
        from dataload import read_images_labels

        k = 10 # k-hot value

        # load train
        # training_images_filepath = './data/train-images-idx3-ubyte/train-images-idx3-ubyte'
        # training_labels_filepath = './data/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
        training_images_filepath = '~/vault/software/mnist/data/train-images-idx3-ubyte/train-images-idx3-ubyte'
        training_labels_filepath = '~/vault/software/mnist/data/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
        x_train, y_train = read_images_labels(training_images_filepath, training_labels_filepath)

        # load test
        # test_images_filepath = './data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
        # test_labels_filepath = './data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
        test_images_filepath = '~/vault/software/mnist/data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
        test_labels_filepath = '~/vault/software/mnist/data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
        x_test, y_test = read_images_labels(test_images_filepath, test_labels_filepath)

        # reformat data for model
        x_train = x_train.transpose() * (1/255)

        # data dimension D
        D = x_train.shape[0]
        assert D==784
        # number of training examples N
        N = x_train.shape[1]

        # reformat data to k-hot format TODO: does numpy have a better way to do this?
        temp_array = np.zeros((k, N))
        for index,val in enumerate(y_train):
            temp_array[val][index] = 1
        y_train = temp_array

        x_test = x_test.transpose() * (1/255)
        y_test = y_test.reshape(1,-1)

        print('MNIST data loaded in.')

        ## initialize network
        network = Network(dims=(784,15,10), activation_funcs = [(sigmoid, sigmoid_prime),(sigmoid, sigmoid_prime)], loss=(mse_loss, mse_loss_prime), cost=cost, seed=1)
        
        # train on data with following parameters
        epochs = 3
        learning_rate = 0.01
        batch_size = 10

        np.set_printoptions(suppress=True, linewidth = 150)

        print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate={learning_rate}')
        network.train(train_data=x_train, labels=y_train, batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, verbose=True)
        print('Training completed.')

        # test performance
        accuracy = network.test(test_data=x_test, test_labels=y_test, verbose=True)
        print(f'Training resulted in network with {accuracy*100 :.4}% accuracy.')

    if iris:
        # iris dataset
        k = 3 # k-hot value

        # csv columns are sepal_length,sepal_width,petal_length,petal_width,species
        iris_dataset = np.genfromtxt('iris.csv', delimiter=',')

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

        ## initialize network
        network = Network(dims=(4,5,3), activation_funcs = [(sigmoid, sigmoid_prime),(sigmoid, sigmoid_prime)], loss=(mse_loss, mse_loss_prime), cost=cost, seed=1)
        
        # train on data with following parameters
        epochs = 300
        learning_rate = 0.1
        batch_size = 5

        np.set_printoptions(suppress=True, linewidth = 150)

        print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate={learning_rate}')
        network.train(train_data=x_train, labels=y_train, batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, verbose=True)
        print('Training completed.')

        # test performance
        accuracy = network.test(test_data=x_test, test_labels=y_test, verbose=True)
        print(f'Training resulted in network with {accuracy*100 :.4}% accuracy.')

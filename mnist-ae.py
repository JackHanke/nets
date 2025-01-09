from models.ann.ann import ArtificialNeuralNetwork, train_ann
from functions.activation_funcs import *
from functions.loss_funcs import *
from functions.optimizers import *
from models.ae.ae import AutoEncoder
from functions.anim_funcs import *
from datasets.mnist.dataload import get_mnist_data
from time import time
import pickle

# creates autoencoder for MNIST
def mnist_ae(path=None):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )
    print('MNIST data loaded in.')
    # NOTE full dataset
    train_data = np.hstack((x_train, x_valid))
    train_labels = np.hstack((x_train, x_valid))

    if path is None:
        learning_rate = 0.05
        epochs = 100
        batch_size = 128

        ae = AutoEncoder(
            dims = (784, 128, 128, 16, 128, 128, 784),
            activation_funcs = [LeakyReLu(), LeakyReLu(), LeakyReLu(), LeakyReLu(), LeakyReLu(), Sigmoid()], 
            loss = MSE(), 
            seed = 1,
            version_num = 0,
            add_noise = True
        )

        # set the optimizer
        optimizer = SGD(
            learning_rate = 0.1,
            weight_decay = 0.99999
        )

        print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate={learning_rate}')
        start = time()
        train_ann(
            model = ae,
            train_data = train_data, 
            train_labels = train_labels,
            valid_data = None,
            valid_labels = None,
            batch_size = batch_size, 
            epochs = epochs,
            optimizer = optimizer,
            verbose = True,
            plot_learning = True
        )
        print(f'Training completed in {((time()-start)/60):.4f} minutes.')
        
        path_str = f'models/ae/saves/mnist_ae_{ae.version_num}.pkl'
        with open(path_str, 'wb') as f:
            pickle.dump(ae, file=f)
        print(f'Model saved at: {path_str}')

    elif path:
        with open(path, 'rb') as f:
            ae = pickle.load(f)

    return ae

if __name__ == '__main__':
    ae = mnist_ae(path=None)
    # ae = mnist_ae(path=f'models/ae/saves/mnist_ae_{0}.pkl')

from models.ann.ann import ArtificialNeuralNetwork
from functions.activation_funcs import *
from functions.loss_funcs import *
from models.vae.vae import VariationalAutoEncoder, train_vae
from functions.anim_funcs import *
from functions.optimizers import *
from datasets.mnist.dataload import get_mnist_data
from time import time
import pickle

# creates variational autoencoder for MNIST
def mnist_vae(path=None):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )
    print('MNIST data loaded in.')

    if path is None:
        # hyperparams
        latent_dim = 8
        reg_weight_update = 0.00001
        epochs = 1000
        batch_size = 128

        encodernet = ArtificialNeuralNetwork(
            dims = (784, 128, 128, 2*latent_dim),
            activation_funcs = [LeakyReLu(), LeakyReLu(), LeakyReLu()], 
            loss = (VAEInternal(latent_dim=latent_dim, reg_weight_update=reg_weight_update)),
            seed = None,
            version_num = 0
        )
        decodernet = ArtificialNeuralNetwork(
            dims = (latent_dim, 128, 128, 784),
            activation_funcs = [LeakyReLu(), LeakyReLu(), Sigmoid()], 
            loss = MSE(), 
            seed = None,
            version_num = 1
        )
        vae = VariationalAutoEncoder(
            encodernet = encodernet,
            decodernet = decodernet
        )

        # set the optimizer
        optimizer = SGD(
            learning_rate = 0.01,
            weight_decay = 1
        )

        print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate={optimizer.learning_rate} with reg_weight_update={reg_weight_update}')
        start = time()
        train_vae(
            model=vae,
            train_data=x_train, 
            train_labels=x_train,
            valid_data=x_valid,
            valid_labels=x_valid,
            batch_size=batch_size, 
            epochs=epochs, 
            optimizer=optimizer,
            verbose=True,
            plot_learning=True
        )
        print(f'Training completed in {((time()-start)/60):.4f} minutes.')
        
        path_str = f'models/vae/saves/mnist_vae_{vae.version_num}.pkl'
        with open(path_str, 'wb') as f:
            pickle.dump(vae, file=f)
        print(f'Model saved at: {path_str}')

    elif path:
        with open(path, 'rb') as f:
            vae = pickle.load(f)

    return vae

if __name__ == '__main__':
    vae = mnist_vae(path=None)
    # vae = mnist_vae(path=f'models/vae/saves/mnist_vae_{0}.pkl')
    
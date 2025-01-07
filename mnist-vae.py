from models.ann.ann import ArtificialNeuralNetwork
from functions.activation_funcs import *
from functions.loss_funcs import *
from models.vae.vae import VariationalAutoEncoder
from functions.anim_funcs import *
from datasets.mnist.dataload import get_mnist_data
from time import time
import pickle

# TODO not done
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
        latent_dim = 15
        reg_weight_update = 0.00001
        learning_rate = 0.01
        epochs = 200
        batch_size = 128
        weight_decay = 1

        encodernet = ArtificialNeuralNetwork(
            dims=(784, 128, 2*latent_dim),
            activation_funcs = [Sigmoid(), TanH()], 
            loss=(VAEInternal(latent_dim=latent_dim, reg_weight_update=reg_weight_update)), # TODO write custom class for this
            seed=None,
            version_num=0
        )
        decodernet = ArtificialNeuralNetwork(
            dims=(latent_dim, 128, 784),
            activation_funcs = [Sigmoid(), Sigmoid()], 
            loss=(MSE()), 
            seed=None,
            version_num=0
        )
        vae = VariationalAutoEncoder(
            encodernet=encodernet,
            decodernet=decodernet
        )

        print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate={learning_rate} with reg_weight_update={reg_weight_update}')
        start = time()
        vae.train(
            train_data=x_train, 
            train_labels=x_train,
            valid_data=x_valid,
            valid_labels=x_valid,
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            weight_decay=weight_decay,
            epochs=epochs, 
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
    
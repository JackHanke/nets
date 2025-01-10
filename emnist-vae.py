from models.ann.ann import ArtificialNeuralNetwork
from functions.activation_funcs import *
from functions.loss_funcs import *
from models.vae.vae import VariationalAutoEncoder, train_vae
from functions.anim_funcs import *
from functions.optimizers import *
from datasets.emnist.dataload import get_emnist_data
from time import time
import pickle

# creates variational autoencoder for EMNIST
def emnist_vae(path=None):
    x_train, y_train = get_emnist_data(path='./datasets/emnist/emnist-byclass.mat')
    print(f'{x_train.shape[1]} rows of EMNIST data loaded in.')

    if path is None:
        # hyperparams
        latent_dim = 8
        reg_weight_update = 0.00001
        epochs = 30
        batch_size = 256

        encodernet = ArtificialNeuralNetwork(
            dims = (784, 256, 128, 2*latent_dim),
            activation_funcs = [TanH(), TanH(), LeakyReLu()], 
            loss = (VAEInternal(latent_dim=latent_dim, reg_weight_update=reg_weight_update)),
            seed = None,
            version_num = 0
        )
        decodernet = ArtificialNeuralNetwork(
            dims = (latent_dim, 128, 256, 784),
            activation_funcs = [TanH(), TanH(), Sigmoid()], 
            loss = MSE(), 
            seed = None,
            version_num = 1
        )
        vae = VariationalAutoEncoder(
            encodernet = encodernet,
            decodernet = decodernet
        )

        # encoder_optimizer = SGD(
        #     learning_rate = 0.01,
        #     weight_decay = 0.9999
        # )

        # decoder_optimizer = SGD(
        #     learning_rate = 0.01,
        #     weight_decay = 0.9999
        # )

        encoder_optimizer = ADAM(
            weights=encodernet.weights,
            biases=encodernet.biases
        )

        decoder_optimizer = ADAM(
            weights=decodernet.weights,
            biases=decodernet.biases
        )

        print(f'Beginning training for {epochs} epochs at batch size {batch_size} with reg_weight_update={reg_weight_update}')
        start = time()
        train_vae(
            model=vae,
            train_data=x_train, 
            train_labels=x_train,
            valid_data=None,
            valid_labels=None,
            batch_size=batch_size, 
            epochs=epochs, 
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            verbose=True,
            plot_learning=True
        )
        print(f'Training completed in {((time()-start)/60):.4f} minutes.')
        
        path_str = f'models/vae/saves/emnist_vae_{vae.version_num}.pkl'
        with open(path_str, 'wb') as f:
            pickle.dump(vae, file=f)
        print(f'Model saved at: {path_str}')

    elif path:
        with open(path, 'rb') as f:
            vae = pickle.load(f)

    return vae

if __name__ == '__main__':
    vae = emnist_vae(path=None)
    # vae = emnist_vae(path=f'models/vae/saves/mnist_vae_{0}.pkl')
    
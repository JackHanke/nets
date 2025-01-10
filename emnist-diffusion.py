from models.ann.ann import ArtificialNeuralNetwork
from models.diffusion.diffusion import Diffusion, train_diff
from functions.activation_funcs import *
from functions.loss_funcs import *
from functions.anim_funcs import *
from functions.optimizers import *
import numpy as np
from time import time
from datasets.emnist.dataload import get_emnist_data
import matplotlib.pyplot as plt
import pickle

# create denoising diffusion model for EMNIST
def emnist_diffusion(path=None):
    if path is None:
        with open(f'datasets/mnist/vae-encoded-emnist.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open(f'datasets/mnist/emnist-ytrain.pkl', 'rb') as f:
            train_labels = pickle.load(f)
        print('EMNIST data loaded in.')

        T, x_dim, y_dim, color_dim, condition_dim = 16, 8, 1, 1, 62
        epochs = 150
        batch_size = 256

        diff = Diffusion(
            dims=(train_data.shape[0]+condition_dim+T, 1000, 500, 500, train_data.shape[0]),
            activation_funcs = [TanH(), TanH(), TanH(), Identity()], 
            loss=(MSE()), 
            seed=None,
            version_num=0,
            T=T,
            x_dim=x_dim,
            y_dim=y_dim,
            color_dim=color_dim,
            condition_dim=condition_dim
        )

        # set the optimizer
        optimizer = SGD(
            learning_rate = 1*(10**(-4)),
            weight_decay = 1
        )

        optimizer = ADAM(
            weights=diff.weights,
            biases=diff.biases
        )

        print(f'Beginning training {diff.num_params()} parameters for {epochs} epochs at batch size {batch_size} at learning rate={optimizer.learning_rate}')
        start = time()
        train_diff(
            model=diff,
            train_data=train_data, 
            train_conditions=train_labels, 
            batch_size=batch_size, 
            epochs=epochs,
            optimizer=optimizer,
            verbose=True,
            plot_learning=True
        )
        print(f'Training completed in {((time()-start)/60):.4f} minutes.')
        
    elif path:
        with open(path, 'rb') as f:
            diff = pickle.load(f)

    return diff

if __name__ == '__main__':
    # ae_path = f'models/ae/saves/mnist_ae_{0}.pkl'
    ae_path = f'models/vae/saves/emnist_vae_{0}.pkl'
    get_and_encode_mnist(ae_path=ae_path)


    # get autoencoder
    # with open(f'models/vae/saves/mnist_vae_{0}.pkl', 'rb') as f:
    #     ae = pickle.load(f)

    diff = mnist_diffusion(path=None)
    # diff = mnist_diffusion(path=f'models/diffusion/saves/mnist_diffusion_{0}.pkl')

    vec_history = diff.gen(condition=0, return_history=True)
    # anim_ims(arr=vec_history, save_path=f'models/diffusion/anim3.gif', fps=8, show=False)

    # encode inference
    history = []
    for vec in vec_history:
        temp = np.reshape(vec, (-1,1))
        # im = ae.decode(activation=temp)
        im = ae.decode(activation=temp)
        im = np.reshape(im, (28, 28))
        history.append(im)

    history += [im for _ in range(12)]

    anim_ims(arr=history, save_path=f'models/diffusion/anim-letter.gif', fps=4, show=False)


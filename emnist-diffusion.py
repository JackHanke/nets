from models.ann.ann import ArtificialNeuralNetwork
from models.diffusion.diffusion import Diffusion, train_diff
from functions.activation_funcs import *
from functions.loss_funcs import *
from functions.anim_funcs import *
from functions.optimizers import *
import numpy as np
from time import time
from datasets.emnist.dataload import get_emnist_data
from preprocessing import get_and_encode_emnist
import matplotlib.pyplot as plt
import pickle

# create denoising diffusion model for EMNIST
def emnist_diffusion(path=None):
    if path is None:
        # train_data, train_labels = get_emnist_data(path=None)
        path_str = f'datasets/emnist/vae-encoded-emnist.pkl'
        with open(path_str, 'rb') as f:
            train_data = pickle.load(f)

        path_str = f'datasets/emnist/emnist-ytrain.pkl'
        with open(path_str, 'rb') as f:
            train_labels = pickle.load(f)
        print(f'EMNIST data loaded in. train_data.shape={train_data.shape} train_labels.shape={train_labels.shape}')

        T, x_dim, y_dim, color_dim, condition_dim = 50, 16, 1, 1, 37
        epochs = 150
        batch_size = 256

        diff = Diffusion(
            dims=(train_data.shape[0]+condition_dim+T, 500, 500, 500, train_data.shape[0]),
            activation_funcs = [LeakyReLu(), LeakyReLu(), LeakyReLu(), Identity()], 
            loss=(MSE()), 
            seed=None,
            version_num=0,
            T=T,
            x_dim=x_dim,
            y_dim=y_dim,
            color_dim=color_dim,
            condition_dim=condition_dim
        )

        optimizer = ADAM(
            weights=diff.weights,
            biases=diff.biases
        )

        print(f'Beginning training {diff.num_params()} parameters for {epochs} epochs at batch size {batch_size}')
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
    # get_and_encode_mnist(ae_path=ae_path)
    # ae_path = f'models/vae/saves/emnist_vae_{0}.pkl'
    # get_and_encode_emnist(ae_path=ae_path)

    with open(f'models/vae/saves/emnist_vae_{0}.pkl', 'rb') as f:
        ae = pickle.load(f)

    # diff = emnist_diffusion(path=None)
    diff = emnist_diffusion(path=f'models/diffusion/saves/emnist_diffusion_{0}.pkl')

    vec_history = diff.gen(condition=2, return_history=True)
    # anim_ims(arr=vec_history, save_path=f'models/diffusion/anim3.gif', fps=8, show=False)

    # encode inference
    history = []
    for vec in vec_history:
        temp = np.reshape(vec, (-1,1))
        # im = ae.decode(activation=temp)
        im = ae.decode(activation=temp)
        im = np.reshape(im, (28, 28))
        history.append(im)

    history += [im for _ in range(24)]

    anim_ims(arr=history, save_path=f'models/diffusion/anim-letter.gif', fps=8, show=False)


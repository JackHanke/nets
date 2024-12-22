from models.ann.ann import ArtificialNeuralNetwork
from models.diffusion.diffusion import Diffusion
from functions.activation_funcs import *
from functions.loss_funcs import *
from functions.anim_funcs import *
import numpy as np
from time import time
from datasets.emnist.dataload import get_data, get_data_small
import matplotlib.pyplot as plt
import pickle

def emnist_diffusion(path=None):
    if path is None:
        # x_train, y_train = get_data()
        x_train, y_train = get_data_small()
        print('EMNIST data loaded in.')

        # NOTE
        T, x_dim, y_dim, color_dim, condition_dim = 8, 28, 28, 1, 26
        diff = Diffusion(
            dims=(784+26, 128, 128, 784+26),
            activation_funcs = [Sigmoid(), Sigmoid(), Sigmoid()], 
            loss=(MSE()), 
            seed=1,
            version_num=0,
            T=T,
            x_dim=y_dim,
            y_dim=y_dim,
            color_dim=color_dim,
            condition_dim=condition_dim
        )

        train_data, train_labels = diff.prep_data_for_diffusion(x=x_train, y=y_train, T=T)

        learning_rate = 0.1
        epochs = 25
        batch_size = T

        print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate={learning_rate}')
        start = time()
        diff.train(
            train_data=train_data, 
            train_labels=train_labels,
            valid_data=None,
            valid_labels=None,
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            weight_decay=1,
            epochs=epochs, 
            verbose=True,
            plot_learning=True
        )
            # weight_decay=(1-(5*learning_rate)/(x_train.shape[1])),
        print(f'Training completed in {((time()-start)/60):.4f} minutes.')
        
        path_str = f'models/diffusion/saves/emnist_diffusion_{diff.version_num}.pkl'
        with open(path_str, 'wb') as f:
            pickle.dump(diff, file=f)
        print(f'Model saved at: {path_str}')

    elif path:
        with open(path, 'rb') as f:
            diff = pickle.load(f)

    return diff

if __name__ == '__main__':
    diff = emnist_diffusion(path=None)
    # diff = emnist_diffusion(path=f'models/diffusion/saves/emnist_diffusion_{0}.pkl')
    # vec = diff.gen(condition=0)

    # TODO delete all this crap
    def make_im_arr(vec, x, y):
        row = []
        for i in range(x):
            col = []
            for j in range(y):
                temp = np.reshape(vec[:-26][:, (x*i)+j], (784,1))
                # input(temp.shape)
                temp = np.reshape(temp, (28,28))
                temp = np.flip(np.rot90(temp, k=3), axis=1)
                col.append(temp)
            row.append(np.vstack(col))
        return np.hstack(row)

    im_history = []

    row = []
    for condition in [0,1,2,3]:
        num_gen = 4
        x_vec = np.random.normal(loc=(1/2), scale=(1/6), size=(784, num_gen))
        condition_vec = np.zeros((26, num_gen))
        if condition is not None: 
            for i in range(num_gen): condition_vec[condition][i] = 1
        x_vec = np.vstack((x_vec, condition_vec))
        row.append(x_vec)
    x_vec = np.hstack(row)

    im = make_im_arr(vec=x_vec, x=4, y=4)
    im_history.append(im)
    for t in range(diff.T): 
        x_vec = diff._forward(activation=x_vec)

        im = make_im_arr(vec=x_vec, x=4, y=4)
        im_history.append(im)

    anim(arr=im_history[:-1], save_path=f'models/diffusion/anim2.gif', fps=4)


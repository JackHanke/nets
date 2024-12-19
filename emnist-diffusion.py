from models.ann.ann import ArtificialNeuralNetwork
from models.diffusion.diffusion import prep_data_for_diffusion, Diffusion
from functions.activation_funcs import *
from functions.loss_funcs import *
from functions.anim_funcs import *
import numpy as np
from time import time
from datasets.emnist.dataload import get_data
import matplotlib.pyplot as plt
import pickle

def emnist_diffusion(diffusion_path=None):
    if diffusion_path is None:
        
        print('MNIST data loaded in.')

        T, x_dim, y_dim, color_dim, condition_dim = 8, 28, 28, 1, 26

        train_data, train_labels = prep_data_for_diffusion(x=x_train, y=y_train, T=T)
        valid_data, valid_labels = prep_data_for_diffusion(x=x_valid, y=y_valid, T=T)

        diff = Diffusion(
            dims=(794, 128, 128, 794),
            activation_funcs = [Sigmoid(), Sigmoid(), Sigmoid()], 
            loss=(MSE()), 
            seed=1,
            version_num=0,
            T=T
        )

        learning_rate = 0.1
        epochs = 50
        batch_size = 3*T

        print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate={learning_rate}')
        start = time()
        diff.train(
            train_data=train_data, 
            train_labels=train_labels, # TODO change!
            valid_data=valid_data,
            valid_labels=valid_labels, # TODO change!link
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            weight_decay=(1-(5*learning_rate)/(x_train.shape[1])),
            epochs=epochs, 
            verbose=True,
            plot_learning=True
        )
        print(f'Training completed in {((time()-start)/60):.4f} minutes.')
        
        path_str = f'models/diffusion/saves/emnist_diffusion_{diff.version_num}.pkl'
        with open(path_str, 'wb') as f:
            pickle.dump(diff, file=f)
        print(f'Model saved at: {path_str}')

    elif diffusion_path:
        with open(diffusion_path, 'rb') as f:
            diff = pickle.load(f)

    return diff

if __name__ == '__main__':
    diff = mnist_diffusion(diffusion_path=None)
    # diff = emnist_diffusion(diffusion_path=f'models/diffusion/saves/emnist_diffusion_{0}.pkl')
    # vec = diff.gen(condition=0)

    def make_im_arr(vec, x, y):
        row = []
        for i in range(x):
            col = []
            for j in range(y):
                temp = np.reshape(x_vec[:-10][:, (x*i)+j], (784,1))
                # input(temp.shape)
                col.append(np.reshape(temp, (28,28)))
            row.append(np.vstack(col))
        return np.hstack(row)

    im_history = []

    row = []
    for condition in [1,7,3,8]:
        num_gen = 4
        x_vec = np.random.normal(loc=(1/2), scale=(1/4), size=(784, num_gen))
        condition_vec = np.zeros((10, num_gen))
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

    anim(arr=im_history[:-1], save_path=f'models/diffusion/anim.gif', fps=4)


from models.ann.ann import ArtificialNeuralNetwork
from models.diffusion.diffusion import Diffusion
from functions.activation_funcs import *
from functions.loss_funcs import *
from functions.anim_funcs import *
import numpy as np
from time import time
import matplotlib.pyplot as plt
import pickle
import sklearn

def moon_diffusion(path=None):
    if path is None:
        train_data, train_labels = sklearn.datasets.make_moons(n_samples=1000)
        train_data = train_data.transpose()
        train_labels = np.eye(2)[train_labels].transpose()

        # plt.scatter(train_data[0], train_data[1], c=train_labels)
        # plt.show()

        valid_data, valid_labels = sklearn.datasets.make_moons(n_samples=1000)
        valid_data = valid_data.transpose()
        valid_labels = np.eye(2)[valid_labels].transpose()

        # plt.scatter(train_data[0], train_data[1])
        # plt.show()

        # transform data to be inside -1,1 for both dimensions

        # xmin, xmax = np.min(train_data[0]), np.max(train_data[0])
        # ymin, ymax = np.min(train_data[1]), np.max(train_data[1])
        # xs = train_data[0]*(1/(xmax-xmin)) - (xmin+1)
        # ys = train_data[1]*(1/(ymax-ymin)) - (ymin+1)
        # train_data = np.vstack((xs, ys))

        # plt.scatter(train_data[0], train_data[1])
        # plt.show()

        print('Moons dataset created')
        
        T, x_dim, y_dim, color_dim, condition_dim = 10, 2, 1, 1, 2
        diff = Diffusion(
            dims=(2+condition_dim+T, 200, 200, 100, 100, 2),
            activation_funcs = [LeakyReLu(), LeakyReLu(), LeakyReLu(), LeakyReLu(), Identity()],
            loss=(MSE()), 
            seed=None,
            version_num=0,
            T=T,
            x_dim=x_dim,
            y_dim=y_dim,
            color_dim=color_dim,
            condition_dim=condition_dim
        )

        learning_rate = 1*(10**(-5))
        epochs = 3500
        batch_size = 50

        print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate={learning_rate}')
        start = time()
        diff.train(
            train_data=train_data, 
            train_conditions=train_labels, 
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            weight_decay=1,
            epochs=epochs,
            valid_data=None,
            valid_conditions=None,
            verbose=True,
            plot_learning=True
        )
        print(f'Training completed in {((time()-start)/60):.4f} minutes.')
        
        path_str = f'models/diffusion/saves/moon_diffusion_{diff.version_num}.pkl'
        with open(path_str, 'wb') as f:
            pickle.dump(diff, file=f)
        print(f'Model saved at: {path_str}')

    elif path:
        with open(path, 'rb') as f:
            diff = pickle.load(f)

    return diff

if __name__ == '__main__':
    # diff = moon_diffusion(path=None)
    diff = moon_diffusion(path=f'models/diffusion/saves/moon_diffusion_{0}.pkl')

    vec_history = diff.gen(condition=None, num_gen=500, return_history=True)

    train_data, _ = sklearn.datasets.make_moons(n_samples=100)
    train_data = train_data.transpose()
    anim_plot(arr=vec_history, save_path=f'models/diffusion/moon-anim.gif', fps=5, show=False, train=train_data)



    
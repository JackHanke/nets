from models.ann.ann import ArtificialNeuralNetwork
from models.diffusion.diffusion import Diffusion
from functions.activation_funcs import *
from functions.loss_funcs import *
from functions.anim_funcs import *
import numpy as np
from time import time
from datasets.mnist.dataload import get_mnist_data
import matplotlib.pyplot as plt
import pickle

def mnist_diffusion(diffusion_path=None):
    if diffusion_path is None:
        x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
            train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
            train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
            test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
            test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
        )
        print('MNIST data loaded in.')

        T, alpha, x_dim, y_dim, color_dim, condition_dim = 256, 0.98, 28, 28, 1, 10
        diff = Diffusion(
            dims=(794, 128, 128, 784),
            activation_funcs = [Sigmoid(), Sigmoid(), Identity()], 
            loss=(MSE()), 
            seed=1,
            version_num=0,
            T=T,
            alpha=alpha,
            x_dim=x_dim,
            y_dim=y_dim,
            color_dim=color_dim,
            condition_dim=condition_dim
        )

        learning_rate = 0.0001
        epochs = 20
        batch_size = 1024

        print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate={learning_rate}')
        start = time()
        diff.train(
            train_data=((2*x_train)-1), 
            train_conditions=y_train, 
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            weight_decay=(1-(5*learning_rate)/(x_train.shape[1])),
            epochs=epochs, 
            verbose=True,
            plot_learning=True
        )
        print(f'Training completed in {((time()-start)/60):.4f} minutes.')
        
        path_str = f'models/diffusion/saves/mnist_diffusion_{diff.version_num}.pkl'
        with open(path_str, 'wb') as f:
            pickle.dump(diff, file=f)
        print(f'Model saved at: {path_str}')

    elif diffusion_path:
        with open(diffusion_path, 'rb') as f:
            diff = pickle.load(f)

    return diff

if __name__ == '__main__':
    diff = mnist_diffusion(diffusion_path=None)
    # diff = mnist_diffusion(diffusion_path=f'models/diffusion/saves/mnist_diffusion_{0}.pkl')
    # vec = diff.gen(condition=0)

    def make_im_arr(vecs, t, x, y):
        row = []
        for i in range(x):
            col = []
            for j in range(y):
                temp = vecs[(x*j) + i][t]
                # rescale to [0, 1]
                temp = (temp+1)/2
                col.append(temp)
            row.append(np.vstack(col))
        return np.hstack(row)

    vecs = []
    for condition in [8,8,8,8]:
        vec_history = diff.gen(condition=condition, return_history=True)
        vecs.append(vec_history)

    im_history = []
    for t in range(diff.T):
        im_history.append(make_im_arr(vecs=vecs, t=t, x=2, y=2))

    # NOTE add pause for final image``
    im_history += [im_history[-1] for _ in range(10)]

    anim(arr=im_history, save_path=f'models/diffusion/anim.gif', fps=15)

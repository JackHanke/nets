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
        # NOTE full dataset
        # train_data = np.hstack((x_train, x_valid))
        # train_labels = np.hstack((y_train, y_valid))

        # NOTE just valid dataset
        train_data = x_valid
        train_labels = y_valid

        # NOTE one datapoint dataset
        # train_data = np.reshape(x_train[:, 0], (-1,1))
        # train_labels = np.reshape(y_train[:, 0], (-1,1))

        # NOTE random dataset for testing purposes
        # train_data = np.random.normal(loc=0, scale=1, size=(784, 60000))
        # train_labels = np.zeros((10, 60000))

        print('MNIST data loaded in.')
        T, x_dim, y_dim, color_dim, condition_dim = 1000, 28, 28, 1, 10
        diff = Diffusion(
            dims=(784+10+1, 784, 784),
            activation_funcs = [TanH(), Identity()], 
            loss=(MSE()), 
            seed=1,
            version_num=0,
            T=T,
            x_dim=x_dim,
            y_dim=y_dim,
            color_dim=color_dim,
            condition_dim=condition_dim
        )

        learning_rate = 0.0001
        epochs = 30
        # batch_size = 1
        batch_size = 128

        print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate={learning_rate}')
        start = time()
        diff.train(
            train_data=((2*train_data)-1), 
            train_conditions=train_labels, 
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            weight_decay=0.99,
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
    for condition in [5,5,5,5]:
        vec_history = diff.gen(condition=condition, return_history=True)
        vecs.append(vec_history)

    im_history = []
    for t in range(diff.T):
        im_history.append(make_im_arr(vecs=vecs, t=t, x=2, y=2))

    # NOTE add pause for final image
    im_history += [im_history[-1] for _ in range(2)]

    anim(arr=im_history, save_path=f'models/diffusion/anim.gif', fps=16)

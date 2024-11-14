from models.ann.ann import ArtificialNeuralNetwork
from functions.activation_funcs import *
from functions.loss_funcs import *
import numpy as np
from time import time
from datasets.mnist.dataload import get_mnist_data
import matplotlib.pyplot as plt
# from models.diffusion.diffusion import Diffusion

import matplotlib.animation as animation

def anim(im_history, save=False):
    fps, nSeconds = 8, 5
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    # fig = plt.figure( figsize=(8,8) )
    a = im_history[0]
    im = plt.imshow(a, vmin=0, vmax=1)
    plt.set_cmap('Grays')
    plt.clim(0,1)
    plt.axis('off')

    def animate_func(i):
        if i % fps == 0: print( '.', end ='' )
        im.set_array(im_history[i])
        return [im]

    anim = animation.FuncAnimation(
                fig, 
                animate_func, 
                frames = (nSeconds * fps),
                interval = (1000 / fps), # in ms
            )
    anim.save('./models/diffusion/diff-anim.gif', fps=fps)

def add_noise(im):
    noise = np.random.normal(loc=0, scale=0.07, size=(28,28))
    result = im + noise
    tobigmask = result > 1
    result[tobigmask] = 1
    tosmallmask = result < 0
    result[tosmallmask] = 0
    return result # TODO is this the right way to do this?

def mnist_benchmark(network, save=False):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )
    print('MNIST data loaded in.')

    im = x_train[:, 0].reshape(28,28)

    im_history = [im]
    for _ in range(500):
        im = add_noise(im)
        im_history.append(im)

    anim(im_history)

if __name__ == '__main__':
    mnist_benchmark(network=None)

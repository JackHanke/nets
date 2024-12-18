from models.ann.ann import ArtificialNeuralNetwork
from models.ae.ae import AutoEncoder
from models.vae.vae import VariationalAutoEncoder
from models.diffusion.diffusion import prep_data_for_diffusion, Diffusion
from functions.activation_funcs import *
from functions.loss_funcs import *
import numpy as np
from time import time
from datasets.mnist.dataload import get_mnist_data
import matplotlib.pyplot as plt
import pickle

import matplotlib.animation as animation

# add 
def add_noise(im, noise, alpha):
    result =  alpha*noise + (1-alpha)*im
    return result

# 
def anim(im_history, save_path):
    fps, nSeconds = 10, len(im_history)//10
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
    anim.save(save_path, fps=fps)

# 
def mnist_noise_anim(network, save_path):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )
    print('MNIST data loaded in.')

    im = x_train[:, 0].reshape(28,28)
    noise = np.random.uniform(low=0, high=1, size=(28,28))
    alpha = 0.07

    im_history = [im]
    for _ in range(2000):
        im = add_noise(im=im, noise=noise, alpha=alpha)
        thing = np.hstack((im, noise)) # NOTE probably delete or figure out how to look better
        im_history.append(thing)

    anim(im_history, save_path=save_path)

# 
def mnist_ae_extrap_anim(ae):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )
    print('MNIST data loaded in.')

    im1 = x_test[:, 0].reshape(-1,1) # NOTE this is an image of a '2'
    im2 = x_test[:, 1].reshape(-1,1) # NOTE this is an image of a '7'
    latent1 = ae.encoder_inference(activation=im1)
    latent2 = ae.encoder_inference(activation=im2)

    latents = []
    alpha = 0.01
    for a in np.arange(start=0, stop=1, step=alpha):
        latent_res = latent1*a + latent2*(1-a)
        latents.append(latent_res)
    
    def make_latent_im(ae, latent):
        gen_im = ae.decoder_inference(activation=latent)
        gen_im = gen_im.reshape(28,28)
        latent = latent.reshape(6,6)
        padding1 = np.zeros((11,6))
        padding2 = np.zeros((28,11))
        latent_im_padded = np.vstack((padding1, latent, padding1))
        latent_im_padded = np.hstack((padding2, latent_im_padded, padding2))
        latent_im_padded = np.vstack((gen_im, latent_im_padded))
        return latent_im_padded

    im_history = []
    for latent in latents:
        im = make_latent_im(ae=ae, latent=latent)
        # thing = np.hstack((im, noise)) # NOTE probably delete or figure out how to look better
        im_history.append(im)

    anim(im_history, save_path=f'models/ae/extrap-anim.gif')

# creates picture of messing with the latent space
def mnist_ae(ae_path=None):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )
    print('MNIST data loaded in.')

    if ae_path is None:
        ae = AutoEncoder(
            dims=(784, 64, 36, 64, 784),
            activation_funcs = [Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid()], 
            loss=(MSE()), 
            seed=1,
            version_num=0    
        )

        learning_rate = 0.1
        epochs = 60
        batch_size = 25

        print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate={learning_rate}')
        start = time()
        ae.train(
            train_data=x_train, 
            train_labels=x_train, # NOTE
            valid_data=x_valid,
            valid_labels=x_valid,
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            weight_decay=(1-(5*learning_rate)/(x_train.shape[1])),
            epochs=epochs, 
            verbose=True,
            plot_learning=True
        )
        print(f'Training completed in {((time()-start)/60):.4f} minutes.')
        
        path_str = f'models/ae/saves/mnist_ae_{ae.version_num}.pkl'
        with open(path_str, 'wb') as f:
            pickle.dump(ae, file=f)
        print(f'Model saved at: {path_str}')

    elif ae_path:
        with open(ae_path, 'rb') as f:
            ae = pickle.load(f)

    return ae

# 
def mess_with_ae_gen(ae, image, save=False):
    np.random.seed(343)
    noise = np.random.uniform(low=0, high=1, size=(6,6))

    # inference = ae._forward(activation=original)
    latent = ae.encoder_inference(activation=image)
    inference = ae.decoder_inference(activation=latent)

    latent_im = latent.reshape(6,6)
    noisy_latent_im = add_noise(im=latent_im, noise=noise, alpha=0.35)
    noisy_inf = ae.decoder_inference(activation=noisy_latent_im.reshape(36,1))

    padding = np.zeros((11,6))
    latent_im_padded = np.vstack((padding, latent_im, padding))
    noisy_latent_im_padded = np.vstack((padding, noisy_latent_im, padding))

    image = original.reshape(28,28)
    inference = inference.reshape(28,28)
    noisy_inf = noisy_inf.reshape(28,28)
    
    seq_array = np.hstack((image, latent_im_padded, inference))
    seq_array2 = np.hstack((image, noisy_latent_im_padded, noisy_inf))
    im = np.vstack((seq_array, seq_array2))
    fig = plt.figure()
    plt.imshow(im, vmin=0, vmax=1)
    plt.set_cmap('Grays')
    plt.clim(0,1)
    plt.axis('off')
    plt.show()
    if save: plt.savefig('models/ae/ae-noisy-seven.png')

# 
def mnist_vae(vae_path):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )
    print('MNIST data loaded in.')

    if vae_path is None:
        # vae = VariationalAutoEncoder(
        #     dims=(784, 64, 36, 64, 784),
        #     activation_funcs = [Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid()], 
        #     loss=(MSE()), 
        #     seed=1,
        #     version_num=0    
        # )

        learning_rate = 0.1
        epochs = 60
        batch_size = 25

        print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate={learning_rate}')
        start = time()
        vae.train(
            train_data=x_train, 
            train_labels=x_train, # NOTE
            valid_data=x_valid,
            valid_labels=x_valid,
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            weight_decay=(1-(5*learning_rate)/(x_train.shape[1])),
            epochs=epochs, 
            verbose=True,
            plot_learning=True
        )
        print(f'Training completed in {((time()-start)/60):.4f} minutes.')
        
        path_str = f'models/vae/saves/mnist_vae_{vae.version_num}.pkl'
        with open(path_str, 'wb') as f:
            pickle.dump(vae, file=f)
        print(f'Model saved at: {path_str}')

    elif vae_path:
        with open(vae_path, 'rb') as f:
            vae = pickle.load(f)

    return vae


def mnist_diffusion(diffusion_path=None):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )
    print('MNIST data loaded in.')

    if diffusion_path is None:
        # TODO explain what T is
        T = 32
        train = prep_data_for_diffusion(x=x_train, y=y_train, T=T)
        valid = prep_data_for_diffusion(x=x_valid, y=y_valid, T=T)
        print(f'Data prepared for diffusion training.')

        diff = Diffusion(
            dims=(794, 128, 64, 128, 794),
            activation_funcs = [Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid()], 
            loss=(MSE()), 
            seed=1,
            version_num=0,
            T=T
        )

        learning_rate = 0.1
        epochs = 16
        batch_size = 2*T

        print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate={learning_rate}')
        start = time()
        diff.train(
            train_data=train, 
            train_labels=train,
            valid_data=valid,
            valid_labels=valid,
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
    # mnist_benchmark(network=None)
    # mnist_ae_demo()
    # ae_path = f'models/ae/saves/mnist_ae_0.pkl'
    # ae = mnist_ae(ae_path=ae_path)
    # mnist_ae_extrap_anim(ae=ae)
    # mess_with_ae_gen(ae=ae, image=x_test[:, 0].reshape(-1,1), save=False)

    diff = mnist_diffusion(diffusion_path=None)
    # diff = mnist_diffusion(diffusion_path=f'models/diffusion/saves/mnist_diffusion_0.pkl')

    # vec = diff.gen(condition=0)
    # print(vec)
    # im = np.reshape()
    for condition in range(10):
        x_vec = np.random.normal(loc=0, scale=1, size=(784,1))
        condition_vec = np.zeros((10,1))
        if condition is not None: condition_vec[condition] = 1
        x_vec = np.vstack((x_vec, condition_vec))
        for t in range(diff.T): 
            x_vec = diff._forward(activation=x_vec)

        a = np.reshape(x_vec[:-10], (28,28))
        im = plt.imshow(a, vmin=0, vmax=1)
        plt.set_cmap('Grays')
        plt.clim(0,1)
        plt.axis('off')
        plt.show()





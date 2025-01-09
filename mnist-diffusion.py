from models.ann.ann import ArtificialNeuralNetwork
from models.diffusion.diffusion import Diffusion, train_diff
from functions.activation_funcs import *
from functions.loss_funcs import *
from functions.anim_funcs import *
from functions.optimizers import *
import numpy as np
from time import time
from datasets.mnist.dataload import get_mnist_data
import matplotlib.pyplot as plt
import pickle

# create denoising diffusion model for MNIST
def mnist_diffusion(path=None):
    if path is None:
        with open(f'datasets/mnist/vae-encoded-mnist.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open(f'datasets/mnist/encoded-mnist-ytrain.pkl', 'rb') as f:
            train_labels = pickle.load(f)
        print('MNIST data loaded in.')

        T, x_dim, y_dim, color_dim, condition_dim = 16, 8, 1, 1, 10
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
    # ae_path = f'models/vae/saves/mnist_vae_{0}.pkl'
    # get_and_encode_mnist(ae_path=ae_path)

    # get autoencoder
    with open(f'models/vae/saves/mnist_vae_{0}.pkl', 'rb') as f:
        ae = pickle.load(f)

    # diff = mnist_diffusion(path=None)
    diff = mnist_diffusion(path=f'models/diffusion/saves/mnist_diffusion_{0}.pkl')

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

    anim_ims(arr=history, save_path=f'models/diffusion/anim3.gif', fps=4, show=False)


    # TODO remove
    # max_lst = [np.max(vec) for vec in vec_history]
    # min_lst = [np.min(vec) for vec in vec_history]
    # plt.scatter([i for i in range(len(max_lst)-1)], [max_lst[i+1]/max_lst[i] for i in range(len(max_lst)-1)], label=f'maxs')
    # plt.scatter([i for i in range(len(min_lst)-1)], [min_lst[i+1]/min_lst[i] for i in range(len(min_lst)-1)], label=f'mins')
    # plt.scatter([i for i in range(len(min_lst)-1)], [diff.alphas[i]**(1/2) for i in range(len(min_lst)-1)], label=f'alphas')
    # # plt.scatter([i for i in range(len(min_lst)-1)], [diff.betas[i] for i in range(len(min_lst)-1)], label=f'betas')
    # plt.legend(loc='upper right')
    # plt.ylabel(f'Max vals in vec history')
    # plt.show()

    # TODO remove
    # x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
    #     train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
    #     train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
    #     test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
    #     test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    # )
    # t = 8
    # train_data_batch = 2*np.reshape(x_train[:, 0], (-1,1))-1
    # noisy_vec, epsilon = diff.get_x_t(train_data_batch, t=t)

    # labels_batch = np.zeros((diff.condition_dim, 1))
    # for i in range(1): labels_batch[5][0] = 1

    # time_vec = np.reshape(np.eye(diff.T)[t-1].transpose(), (-1,1))

    # # print(noisy_vec.shape)
    # # print(labels_batch.shape)
    # # input(time_vec.shape)

    # vec = np.vstack((noisy_vec, labels_batch, time_vec))


    # tl = np.reshape(noisy_vec, (28, 28))
    # tr = np.reshape(epsilon, (28, 28))
    # br = np.reshape(diff._forward(activation=vec), (28, 28))
    # bl = np.square(br-tr)
    # b = np.hstack((bl, br))
    # top = np.hstack((tl, tr))
    # im = np.vstack((top, b))
    # im = plt.imshow(im, vmin=0, vmax=1)
    # plt.set_cmap('Grays')
    # plt.clim(-1, 1)
    # plt.axis('off')
    
    # title_str = ''
    # title_str += f't={t} '
    # #  {sqrt(self.alpha**t):.1f} {sqrt(1-self.alpha**t):.1f} '
    # # title_str += f'tdb = {np.max(train_data_batch):.1f} {np.min(train_data_batch):.1f} '
    # title_str += f'nvec = ({np.min(noisy_vec):.1f}, {np.max(noisy_vec):.1f}) '
    # title_str += f'ep = ({np.min(epsilon):.1f}, {np.max(epsilon):.1f}) '
    # title_str += f'pred = ({np.min(br):.1f}, {np.max(br):.1f}) '
    # title_str += f'cost = ({np.min(bl):.1f}, {np.max(bl):.1f}) '
    # plt.title(title_str)
    # plt.show()

    # anim(arr=vec_history, save_path=f'models/diffusion/anim2.gif', fps=16)

    # def make_im_arr(vecs, t, x, y):
    #     row = []
    #     for i in range(x):
    #         col = []
    #         for j in range(y):
    #             temp = vecs[(x*j) + i][t]
    #             # rescale to [0, 1]
    #             temp = (temp+1)/2
    #             col.append(temp)
    #         row.append(np.vstack(col))
    #     return np.hstack(row)

    # vecs = []
    # for condition in [5,5,5,5]:
    #     vec_history = diff.gen(condition=condition, return_history=True)
    #     vecs.append(vec_history)

    # im_history = []
    # for t in range(diff.T):
    #     im_history.append(make_im_arr(vecs=vecs, t=t, x=2, y=2))

    # # NOTE add pause for final image
    # im_history += [im_history[-1] for _ in range(2)]

    # anim(arr=im_history, save_path=f'models/diffusion/anim.gif', fps=16)

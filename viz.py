from datasets.mnist.dataload import get_mnist_data
from functions.anim_funcs import *
import pickle

# TODO clean up

def add_noise(im, noise, alpha):
    result =  alpha*noise + (1-alpha)*im
    return result

def visualize_input_output(ae):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )
    print('MNIST data loaded in.')
    input_im = np.reshape(x_train[:, 0], (28*28,1))
    # noise = np.random.normal(loc=(1/2), scale=(1/6), size=im.shape)
    # input_im = (5/7)*im + (2/7)*noise
    # output_im = ae._forward(activation=input_im)

    output_im = ae._forward(activation=input_im)

    input_im = np.reshape(input_im, (28,28))
    output_im = np.reshape(output_im, (28,28))

    padding1 = np.zeros((28, 2))
    temp = np.hstack((input_im, padding1, output_im))
    padding2 = np.zeros((16,58))
    im = np.vstack((padding2, temp, padding2))

    fig = plt.figure()
    plt.imshow(im, vmin=0, vmax=1)
    plt.set_cmap('Greys')
    plt.clim(0,1)
    plt.axis('off')
    plt.show()
    plt.savefig('models/ae/ae-input-output.png')


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

# animate journey between laten representation of a '2' and a '7'
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

# 
def mess_with_ae_gen(ae, image, save=False):
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

if __name__ == '__main__':
    path = f'models/ae/saves/mnist_ae_{0}.pkl'
    with open(path, 'rb') as f:
        ae = pickle.load(f)

    visualize_input_output(ae=ae)

    # mnist_ae_extrap_anim(ae=ae)

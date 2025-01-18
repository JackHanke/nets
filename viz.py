from datasets.mnist.dataload import get_mnist_data
from datasets.emnist.dataload import get_emnist_data
from functions.anim_funcs import *
import pickle

# Various animation and visualization scripts
# TODO clean up

# 
def add_noise(im, noise, alpha):
    result =  alpha*noise + (1-alpha)*im
    return result

# visualizes multiple input, output pairs of a given autoencoder
def visualize_input_output(ae):
    # x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
    #     train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
    #     train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
    #     test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
    #     test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    # )

    x_train, y_train = get_emnist_data(path='./datasets/emnist/emnist-letters.mat')


    print('MNIST data loaded in.')
    input_im_ttl = np.reshape(x_train[:, 0], (28*28,1))
    input_im_ttr = np.reshape(x_train[:, 1], (28*28,1))
    input_im_tl = np.reshape(x_train[:, 2], (28*28,1))
    input_im_tr = np.reshape(x_train[:, 3], (28*28,1))
    input_im_bl = np.reshape(x_train[:, 4], (28*28,1))
    input_im_br = np.reshape(x_train[:, 5], (28*28,1))
    input_im_bbl = np.reshape(x_train[:, 6], (28*28,1))
    input_im_bbr = np.reshape(x_train[:, 7], (28*28,1))
    # noise = np.random.normal(loc=(1/2), scale=(1/6), size=im.shape)
    # input_im = (5/7)*im + (2/7)*noise
    # output_im = ae._forward(activation=input_im)

    output_im_ttl = ae._forward(activation=input_im_ttl)
    output_im_ttr = ae._forward(activation=input_im_ttr)
    output_im_tl = ae._forward(activation=input_im_tl)
    output_im_tr = ae._forward(activation=input_im_tr)
    output_im_bl = ae._forward(activation=input_im_bl)
    output_im_br = ae._forward(activation=input_im_br)
    output_im_bbl = ae._forward(activation=input_im_bbl)
    output_im_bbr = ae._forward(activation=input_im_bbr)

    input_im_ttl = np.reshape(input_im_ttl, (28,28))
    input_im_ttr = np.reshape(input_im_ttr, (28,28))
    input_im_tl = np.reshape(input_im_tl, (28,28))
    input_im_tr = np.reshape(input_im_tr, (28,28))
    input_im_bl = np.reshape(input_im_bl, (28,28))
    input_im_br = np.reshape(input_im_br, (28,28))
    input_im_bbl = np.reshape(input_im_bbl, (28,28))
    input_im_bbr = np.reshape(input_im_bbr, (28,28))

    output_im_ttl = np.reshape(output_im_ttl, (28,28))
    output_im_ttr = np.reshape(output_im_ttr, (28,28))
    output_im_tl = np.reshape(output_im_tl, (28,28))
    output_im_tr = np.reshape(output_im_tr, (28,28))
    output_im_bl = np.reshape(output_im_bl, (28,28))
    output_im_br = np.reshape(output_im_br, (28,28))
    output_im_bbl = np.reshape(output_im_bbl, (28,28))
    output_im_bbr = np.reshape(output_im_bbr, (28,28))

    # padding = np.zeros((28, 28*4))
    temp_tt = np.hstack((input_im_ttl, output_im_ttl, input_im_ttr, output_im_ttr))
    temp_t = np.hstack((input_im_tl, output_im_tl, input_im_tr, output_im_tr))
    temp_b = np.hstack((input_im_bl, output_im_bl, input_im_br, output_im_br))
    temp_bb = np.hstack((input_im_bbl, output_im_bbl, input_im_bbr, output_im_bbr))
    # padding2 = np.zeros((16,58))
    im = np.vstack((temp_tt, temp_t, temp_b, temp_bb))

    fig = plt.figure()
    plt.imshow(im, vmin=0, vmax=1)
    plt.set_cmap('Greys')
    plt.clim(0,1)
    plt.axis('off')
    plt.show()
    plt.savefig('models/ae/ae-input-output.png')

# animate the transition from data manifold to random data point in data space
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

# animate journey between laten representation of a '2' and a '7', comparing between AE and VAE
def mnist_ae_extrap_anim(ae, vae):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )
    print('MNIST data loaded in.')

    im1 = x_test[:, 0].reshape(-1,1) # NOTE this is an image of a '2'
    im2 = x_test[:, 1].reshape(-1,1) # NOTE this is an image of a '7'
    latent1_ae, latent1_vae = ae.encode(activation=im1), vae.encodernet._forward(activation=im1)
    latent2_ae, latent2_vae = ae.encode(activation=im2), vae.encodernet._forward(activation=im2)

    latents_ae, latents_vae = [], []
    alpha = 0.01
    for a in np.arange(start=0, stop=1, step=alpha):
        latent_res_ae = latent1_ae*a + latent2_ae*(1-a)
        latent_res_vae = latent1_vae*a + latent2_vae*(1-a)
        latents_ae.append(latent_res_ae)
        latents_vae.append(latent_res_vae)
    
    def make_latent_im(ae, latent_ae, vae, latent_vae):
        gen_im_ae = ae.decode(activation=latent_ae)
        gen_im_ae = gen_im_ae.reshape(28,28)
        gen_im_vae = vae.decode(activation=latent_vae[:8])
        gen_im_vae = gen_im_vae.reshape(28,28)

        latent_ae = latent_ae.reshape(4, 4)
        latent_vae = latent_vae.reshape(4, 4)

        padding0 = np.zeros((28,2))
        padding1 = np.zeros((4,28+2+28))
        padding2 = np.zeros((4,12))
        padding3 = np.zeros((4,2))

        final_im_gen = np.hstack((gen_im_ae, padding0, gen_im_vae))
        final_im_latent = np.hstack((padding2, latent_ae, padding2, padding3, padding2, latent_vae, padding2))
        final_im = np.vstack((final_im_gen, padding1, final_im_latent, padding1))
        return final_im

    im_history = []
    for latent_ae, latent_vae in zip(latents_ae, latents_vae):
        im = make_latent_im(ae=ae, latent_ae=latent_ae, vae=vae, latent_vae=latent_vae)
        # thing = np.hstack((im, noise)) # NOTE probably delete or figure out how to look better
        im_history.append(im)

    anim_ims(im_history, save_path=f'models/ae/extrap-anim.gif')

# 
def mess_with_ae_gen(ae, image, save=False):
    noise = np.random.uniform(low=0, high=1, size=(6,6))

    # inference = ae._forward(activation=original)
    latent = ae.encode(activation=image)
    inference = ae.decode(activation=latent)

    latent_im = latent.reshape(6,6)
    noisy_latent_im = add_noise(im=latent_im, noise=noise, alpha=0.35)
    noisy_inf = ae.decode(activation=noisy_latent_im.reshape(36,1))

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

def draw_sentence(diff, ae, sentence_string):
    letter_map = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6,
        'H': 7,
        'I': 8,
        'J': 9,
        'K': 10,
        'L': 11,
        'M': 12,
        'N': 13,
        'O': 14,
        'P': 15,
        'Q': 16,
        'R': 17,
        'S': 18,
        'T': 19,
        'U': 20,
        'V': 21,
        'W': 22,
        'X': 23,
        'Y': 24,
        'Z': 25,
        'a': 26,
        'b': 27,
        'c': 2,
        'd': 28,
        'e': 29,
        'f': 30,
        'g': 31,
        'h': 32,
        'i': 8,
        'j': 9,
        'k': 10,
        'l': 11,
        'm': 12,
        'n': 33,
        'o': 14,
        'p': 15,
        'q': 34,
        'r': 35,
        's': 18,
        't': 36,
        'u': 20,
        'v': 21,
        'w': 22,
        'x': 23,
        'y': 24,
        'z': 25,
    }

    # preprocess
    max_length = max([len(word) for word in sentence_string.split()])
    padding_nums = [max_length - len(word) for word in sentence_string.split()]

    # generate images
    full_history = [[]]
    row_index = 0
    for letter in sentence_string:
        if letter == ' ':
            row_index += 1
            full_history.append([])
        else:
            if letter == '_':
                vec_history = None
            else:
                letter_num = letter_map[letter]
                vec_history = diff.gen(condition=letter_num, return_history=True)
            full_history[row_index].append(vec_history)

    # make frames for animation
    anim_frames = []
    for t in range(diff.T):
        frame = []
        for row_num, row in enumerate(full_history):
            row_im = []
            for letter_hist in row:
                if letter_hist is None:
                    im = np.zeros((28,28))
                else:
                    im = vae.decode(activation=letter_hist[t].transpose())
                    im = np.reshape(im, (28, 28))
                    im = np.flip(np.rot90(im, k=3), axis=1)
                row_im.append(im)
            # add row padding if there is need for padding
            if padding_nums[row_num] != 0:
                padding = np.zeros((28, 14*padding_nums[row_num]))
                # print(padding.shape)
                # print(row_im.shape)
                # frame.append(np.hstack((padding, row_im, padding)))
                row_im = [padding] + row_im + [padding]
            frame.append(np.hstack(row_im))

        anim_frames.append(np.vstack(frame))

    anim_frames += [anim_frames[-1] for _ in range(48)]

    anim_ims(arr=anim_frames, save_path=f'models/diffusion/{sentence_string}.gif', fps=8, show=False)


    '''
    I_love you_Kim
    Neural_Networks from_Scratch
    Hello Hankes
    Hello Linkedin
    Hey Dan
    Hey Horgans
    '''

if __name__ == '__main__':
    # path = f'models/ae/saves/mnist_ae_{0}.pkl'
    # with open(path, 'rb') as f:
    #     ae = pickle.load(f)

    # visualize_input_output(ae=vae)
    # mnist_ae_extrap_anim(ae=ae, vae=vae)

    path = f'models/vae/saves/emnist_vae_{0}.pkl'
    with open(path, 'rb') as f:
        vae = pickle.load(f)

    path = f'models/diffusion/saves/emnist_diffusion_{0}.pkl'
    with open(path, 'rb') as f:
        diff = pickle.load(f)

    draw_sentence(diff=diff, ae=vae, sentence_string=f'I_love you_Kim')

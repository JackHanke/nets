import numpy as np
import scipy.io
import pickle
# various preprocessing scrips for MNIST

# prepares dataset for denoising autoencoder
def noise_dataset(data, T):
    noisy_data = []
    for _ in range(T+1):
        noise = np.random.normal(loc=(1/2), scale=(1/6), size=data.shape)
        data_fragment = data + noise
        noisy_data.append(data_fragment)
    true_data = np.hstack([data for _ in range(T+1)])
    noisy_data = np.hstack(noisy_data)
    return noisy_data, true_data

def noise_dataset_alt(data, T):
    noise = np.random.normal(loc=(1/2), scale=(1/6), size=data.shape)
    noisy_data = []
    for t in range(T+1):
        data_fragment = ((T-t)/T)*data + (t/T) * noise
        noisy_data.append(data_fragment)
    true_data = np.hstack([data for _ in range(T+1)])
    noisy_data = np.hstack(noisy_data)
    return noisy_data, true_data

def get_and_encode_mnist(ae_path):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )
    train_data = ((2*x_train)-1)
    path_str = f'datasets/mnist/mnist-xtrain.pkl'
    with open(path_str, 'wb') as f:
        pickle.dump(train_data, file=f)

    with open(ae_path, 'rb') as f:
        ae = pickle.load(f)

    # encoded_x_train = ae.encode(activation=x_train)
    encoded_x_train = ae.encode(activation=x_train)

    path_str = f'datasets/mnist/vae-encoded-mnist.pkl'
    with open(path_str, 'wb') as f:
        pickle.dump(encoded_x_train, file=f)

    path_str = f'datasets/mnist/encoded-mnist-ytrain.pkl'
    with open(path_str, 'wb') as f:
        pickle.dump(y_train, file=f)

    print(f'Data encoded.')

def get_emnist_data(path=None):
    # original = pd.read_csv(f'./datasets/emnist/emnist-letters-train.csv').to_numpy().transpose()
    # data = original[1:]/255 
    # labels = original[0]
    # conds = np.zeros((26, data.shape[1]))
    # for ind, label in enumerate(labels):
    #     conds[label-1][ind] = 1
    # print(f'Data shape loaded in {data.shape}')

    path_str = f'datasets/emnist/emnist_train.pkl'
    with open(path_str, 'rb') as f:
        x_train = pickle.load(f)

    path_str = f'datasets/emnist/emnist_train_labels.pkl'
    with open(path_str, 'rb') as f:
        y_train = pickle.load(f)

    return x_train, y_train

def get_and_encode_emnist(ae_path):
    x_train, y_train = get_emnist_data(path='./datasets/emnist/emnist-letters.mat')



    path_str = f'datasets/emnist/emnist-xtrain.pkl'
    with open(path_str, 'wb') as f:
        pickle.dump(x_train, file=f)

    with open(ae_path, 'rb') as f:
        ae = pickle.load(f)

    # encoded_x_train = ae.encode(activation=x_train)
    encoded_x_train = ae.encode(activation=x_train)

    path_str = f'datasets/emnist/vae-encoded-emnist.pkl'
    with open(path_str, 'wb') as f:
        pickle.dump(encoded_x_train, file=f)

    path_str = f'datasets/emnist/emnist-ytrain.pkl'
    with open(path_str, 'wb') as f:
        pickle.dump(y_train, file=f)

    print(f'Data encoded.')



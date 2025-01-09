import numpy as np

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

from models.ann.ann import ArtificialNeuralNetwork
from functions.activation_funcs import *
from functions.loss_funcs import *
from models.ae.ae import AutoEncoder
from functions.anim_funcs import *
from datasets.mnist.dataload import get_mnist_data
from time import time
import pickle

# prepares dataset for denoising autoencoder
# 

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

# creates autoencoder for MNIST
def mnist_ae(path=None):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )
    print('MNIST data loaded in.')
    # NOTE full dataset
    # train_data = np.hstack((x_train, x_valid))
    # train_labels = np.hstack((x_train, x_valid))

    # NOTE random dataset for testing
    # np.random.seed(1)
    # x_train = np.random.normal(loc=0, scale=1, size=(784, 48000))
    # x_valid = np.random.normal(loc=0, scale=1, size=(784, 12000))

    # NOTE dataset for denoising, just valid
    train_data, train_labels = noise_dataset(data=x_valid, T=10)
    print(f'Noisy dataset made')

    if path is None:
        ae = AutoEncoder(
            dims=(784, 128, 128, 784),
            activation_funcs = [Sigmoid(), Sigmoid(), Sigmoid()], 
            loss=(MSE()), 
            seed=1,
            version_num=0,
            add_noise=True
        )

        learning_rate = 0.001
        epochs = 100
        batch_size = 128

        print(f'Beginning training for {epochs} epochs at batch size {batch_size} at learning rate={learning_rate}')
        start = time()
        ae.train(
            train_data=train_data, 
            train_labels=train_labels,
            valid_data=None,
            valid_labels=None,
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            weight_decay=(1-(5*learning_rate)/(train_data.shape[1])),
            epochs=epochs, 
            verbose=True,
            plot_learning=True
        )
        print(f'Training completed in {((time()-start)/60):.4f} minutes.')
        
        path_str = f'models/ae/saves/mnist_ae_{ae.version_num}.pkl'
        with open(path_str, 'wb') as f:
            pickle.dump(ae, file=f)
        print(f'Model saved at: {path_str}')

    elif path:
        with open(path, 'rb') as f:
            ae = pickle.load(f)

    return ae

if __name__ == '__main__':
    ae = mnist_ae(path=None)
    # ae = mnist_ae(path=f'models/ae/saves/mnist_ae_{0}.pkl')





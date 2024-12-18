from models.ann.ann import ArtificialNeuralNetwork
from functions.activation_funcs import *
from functions.loss_funcs import *
from models.vae.vae import VariationalAutoEncoder
from functions.anim_funcs import *
from datasets.mnist.dataload import get_mnist_data

# TODO not done
# creates variational autoencoder for MNIST
def mnist_vae(vae_path=None):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )
    print('MNIST data loaded in.')

    if vae_path is None:
        vae = VariationalAutoEncoder(
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
        vae.train(
            train_data=x_train, 
            train_labels=x_train,
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

if __name__ == '__main__':
    vae = mnist_vae(path=None)
    # vae = mnist_vae(path=f'models/vae/saves/mnist_vae_{0}.pkl')
    
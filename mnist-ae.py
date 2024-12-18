from models.ann.ann import ArtificialNeuralNetwork
from functions.activation_funcs import *
from functions.loss_funcs import *
from models.ae.ae import AutoEncoder
from functions.anim_funcs import *
from datasets.mnist.dataload import get_mnist_data


# creates autoencoder for MNIST
def mnist_ae(path=None):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_mnist_data(
        train_im_path='./datasets/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte',
        train_labels_path='./datasets/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        test_im_path='./datasets/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        test_labels_path='./datasets/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    )
    print('MNIST data loaded in.')

    if path is None:
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

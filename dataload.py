# from Hojjat Khodabakhsh at https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook

import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
np.set_printoptions(suppress=True,precision=3, linewidth = 150)

def read_images_labels(images_filepath, labels_filepath):        
    labels = []
    with open(labels_filepath, 'rb') as file: #read 
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())        
    
    images = []
    with open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())        

    for i in range(size):
        images.append([0] * rows * cols)

    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        # img = img.reshape(28, 28)
        images[i][:] = img            
    
    return np.array(images), np.array(labels)
            
if __name__ == '__main__':
    # load train
    training_images_filepath = './data/train-images-idx3-ubyte/train-images-idx3-ubyte'
    training_labels_filepath = './data/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    x_train, y_train = read_images_labels(training_images_filepath, training_labels_filepath)

    # load test
    test_images_filepath = './data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    test_labels_filepath = './data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    x_test, y_test = read_images_labels(test_images_filepath, test_labels_filepath)

    example = np.array(x_train[50145]).reshape(28,28)/255
    print(example)
    print(y_train[50145])
    
    x_train = x_train.transpose() / 255
    y_train = y_train.reshape(1,-1)
    x_test = x_test.transpose() / 255
    y_test = y_test.reshape(1,-1)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)


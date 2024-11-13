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

# get data 
def get_mnist_data(train_im_path, train_labels_path, test_im_path, test_labels_path):
    print(f'Loading MNIST dataset.')
    # load train
    training_images_filepath = train_im_path
    training_labels_filepath = train_labels_path
    x_train, y_train = read_images_labels(training_images_filepath, training_labels_filepath)
    # load test
    test_images_filepath = test_im_path
    test_labels_filepath = test_labels_path
    x_test, y_test = read_images_labels(test_images_filepath, test_labels_filepath)

    # number of training examples N, data dimension D
    N, D = x_train.shape
    true_train_N = int(N*0.8)
    # validation split
    x_true_train, x_valid = x_train[:true_train_N], x_train[true_train_N:]
    # reformat data for model
    x_true_train = x_true_train.transpose() * (1/255)
    x_valid = x_valid.transpose() * (1/255)

    # reformat data to k-hot format TODO: does numpy have a better way to do this?
    temp_array = np.zeros((N, 10))
    for index, val in enumerate(y_train):
        temp_array[index][val] = 1
    y_train = temp_array
    # validation split 
    y_true_train, y_valid = y_train[:true_train_N], y_train[true_train_N:]
    y_true_train, y_valid = y_true_train.transpose(), y_valid.transpose()
    x_test = x_test.transpose() * (1/255)
    y_test = y_test.reshape(1,-1)

    return x_true_train, y_true_train, x_valid, y_valid, x_test, y_test


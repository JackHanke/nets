# this code was created following the lecture notes found here: https://sgfin.github.io/files/notes/CS321_Grosse_Lecture_Notes.pdf
import numpy as np
from dataload import read_images_labels

# numpy settings
np.random.seed(1)
np.set_printoptions(suppress=True,precision=3, linewidth = 150)

# load train
training_images_filepath = './data/train-images-idx3-ubyte/train-images-idx3-ubyte'
training_labels_filepath = './data/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
x_train, y_train = read_images_labels(training_images_filepath, training_labels_filepath)

# load test
test_images_filepath = './data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
test_labels_filepath = './data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
x_test, y_test = read_images_labels(test_images_filepath, test_labels_filepath)

# reformat for 
x_train = x_train.transpose()
y_train = y_train.reshape(1,-1)
x_test = x_test.transpose()
y_test = y_test.reshape(1,-1)

# data dimension D
D = x_train.shape[0]
assert D==784
# number of training examples N
N = x_train.shape[1]

# architecture hyperparameters
a = 15 # width of hidden layer
k = 10 # k-hot value

### instatiate model architecture

# instatiate first matrix of weights
W_1 = np.random.normal(loc=0, scale=1, size=(a,D))

# instatiate first bias vector
b_1 = np.random.normal(loc=0, scale=1, size=(a,1))
ones_1 = np.ones((1,N))

# instatiate second set of weights
W_2 = np.random.normal(loc=0, scale=1, size=(k,a))

# instatiate second set of weights
b_2 = np.random.normal(loc=0, scale=1, size=(k,1))
ones_2 = np.ones((1,N))

def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_prime(x): return sigmoid(x) * sigmoid(1-x)
# implementation of softmax from: https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python 
def softmax(x): 
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

### training

# training hyperparamters
alpha = 0.05 # learning rate
epochs = 5 # number of epochs

# forward pass
Z_1 = W_1.dot(x_train) + b_1.dot(ones_1)
H = sigmoid(Z_1) 
Z_2 = W_2.dot(H) + b_2.dot(ones_2)
Y = sigmoid(Z_2)

def loss(activation, label): return 0.5*(activation-label)**2

def cost(loss_vec): return np.average(loss_vec)

for epoch in range(epochs+1):
    # backprop
    delta_2 = np.multiply((Y-y_train), sigmoid_prime(Z_2))
    delta_1 = np.multiply(W_2.transpose().dot(delta_2), sigmoid_prime(Z_1))

    # FULL gradient descent TODO: stochastic
    # make consistent layer labelling! Nielsen and Gross used different indexing for layer number
    W_1 = W_1 - alpha*delta_1.dot(x_train.transpose())
    b_1 = b_1 - alpha*np.reshape(np.mean(delta_1, axis=1), (-1,1))

    W_2 = W_2 - alpha*delta_2.dot(H.transpose())
    b_2 = b_2 - alpha*np.reshape(np.mean(delta_2, axis=1), (-1,1))

    Z_1 = W_1.dot(x_train) + b_1.dot(ones_1)
    H = sigmoid(Z_1) 
    Z_2 = W_2.dot(H) + b_2.dot(ones_2)
    Y = sigmoid(Z_2)
    loss_val = loss(Y, y_train)
    cost_val = cost(loss_val)
    # if epoch % (epochs//5) == 0: print(f"Cost at epoch {epoch}: {cost_val}")
    print(f"Cost at epoch {epoch}: {cost_val}")


### testing
# forward pass on test data
predictions = np.argmax(sigmoid(W_2.dot(W_1.dot(x_test) + b_1.dot(ones_1)) + b_2.dot(ones_2)), axis=0)


num_wrong = np.sum(np.not_equal(y_test, predictions))
test_size = y_test.shape[1]
print(f"{num_wrong} of {test_size} \nAccuracy: {1 - (num_wrong/test_size)}")








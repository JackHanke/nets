# this code was created following the lecture notes found here: https://sgfin.github.io/files/notes/CS321_Grosse_Lecture_Notes.pdf
import numpy as np
# numpy settings
np.random.seed(1)
np.set_printoptions(suppress=True,precision=3 )

# data & data values
D = 5
# assert D==728
N = 2
# X = 
test_x = np.random.rand(D,2)
print(test_x)
test_y = np.array([[0,0],
                   [0,0],
                   [1,0],
                   [0,0],
                   [0,0],
                   [0,1],
                   [0,0],
                   [0,0],
                   [0,0],
                   [0,0]])

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

# forward pass

Z_1 = W_1.dot(test_x) + b_1.dot(ones_1)
H = sigmoid(Z_1) 
Z_2 = W_2.dot(H) + b_2.dot(ones_2)
Y = sigmoid(Z_2)

def loss(activation, label): return 0.5*(activation-label)**2

def cost(loss_vec): return np.average(loss_vec)
# loss_val = loss(Y, test_y)
# cost_val = cost(loss_val)
# print(cost_val)

# learning rate
alpha = 0.053

epochs = 3000
for epoch in range(epochs+1):
    # backprop
    delta_2 = np.multiply((Y-test_y), sigmoid_prime(Z_2))
    delta_1 = np.multiply(W_2.transpose().dot(delta_2), sigmoid_prime(Z_1))

    # FULL gradient descent TODO: stochastic
    # make consistent layer labelling! Nielsen and Gross used different indexing for layer number
    # print(delta_1)
    # print(delta_1.shape)
    # print(test_x.transpose())
    # print(test_x.transpose().shape)
    # print(delta_1.dot(test_x.transpose()))
    # print(delta_1.dot(test_x.transpose()).shape)
    W_1 = W_1 - alpha*delta_1.dot(test_x.transpose())
    b_1 = b_1 - alpha*np.reshape(np.mean(delta_1, axis=1), (-1,1))

    W_2 = W_2 - alpha*delta_2.dot(H.transpose())
    b_2 = b_2 - alpha*np.reshape(np.mean(delta_2, axis=1), (-1,1))

    Z_1 = W_1.dot(test_x) + b_1.dot(ones_1)
    H = sigmoid(Z_1) 
    Z_2 = W_2.dot(H) + b_2.dot(ones_2)
    Y = sigmoid(Z_2)
    loss_val = loss(Y, test_y)
    cost_val = cost(loss_val)
    if epoch % 100 == 0: print(f"Cost at epoch {epoch}: {cost_val}")

# print(f"Loss:")
# print(loss_val)
print(f"Actual:")
print(test_y)
print(f"Predicted:")
print(Y)



# this code was created following the lecture notes found here: https://sgfin.github.io/files/notes/CS321_Grosse_Lecture_Notes.pdf
import numpy as np

# data & data values
D = 728
assert D==728
N = 1
# X = 
test_x = np.random.rand(728,1)

# architecture hyperparameters

a = 256 # width of hidden layer
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
# implementation of softmax from: https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python 
def softmax(x): 
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# model computation
# log sum exp routine for first activation?

H = softmax(W_1.dot(test_x) + b_1.dot(ones_1)) 

Y = softmax(W_2.dot(H) + b_2.dot(ones_2))


### instatiate loss and cost

# create targets (aka labels)
# targets = 
test_target = np.zeros((10,1))
test_target[5][0] = 1
print(test_target)

# pick loss, in this case cross-entropy
def loss(vec_y,vec_t): return -1*(np.transpose(vec_t)).dot(np.log(vec_y))

# the cost is the average of the loss
def cost(l): return np.average(l)

computed_loss = loss(Y,test_target)
print(f"loss: {computed_loss}")
print(f"cost: {cost(computed_loss)}")

# cost_val = cost(loss(Y,test_target))

# update step hyperparameters
alpha = 0.01

### define update step



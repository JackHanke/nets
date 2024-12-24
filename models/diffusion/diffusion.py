from ..ann.ann import ArtificialNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
from time import time

# NOTE citations
# This class was written after reading the following papers 
#   Ho et al.: https://arxiv.org/pdf/2006.11239
#   Calvin Luo: https://arxiv.org/pdf/2208.11970 
# as well as the YouTube video by Algorithmic Simplicity: https://www.youtube.com/watch?v=zc5NTeJbk-k
class Diffusion(ArtificialNeuralNetwork):
    def __init__(self, T, alpha, x_dim, y_dim, color_dim, condition_dim, pred_data=True, **kwargs):
        super(Diffusion, self).__init__(**kwargs)
        self.T = T # the number of diffusion steps
        self.alpha = alpha # diffusion parameter
        self.beta = 1-alpha # alternate diffusion parameter
        self.x_dim = x_dim # dimension of horizontal pixels in image
        self.y_dim = y_dim # dimension of vertical pixels in image
        self.color_dim = color_dim # number of color channels 
        self.tot_data_dim = self.x_dim*self.y_dim*self.color_dim
        self.condition_dim = condition_dim # dimension of condition vector
        self.tot_dim = self.tot_data_dim + self.condition_dim
        self.pred_data = pred_data # boolean on whether the neural network should predict the source image or the source noise
        # TODO delete below
        self.loc = 1/2 # mean of noise
        self.scale = 1/4 # standard deviation of noise

    # generate num_gen image(s) with a given condition(index for one-hot), 
    def gen(self, condition=None, num_gen=1, return_history=False):
        im_history = []
        # initialize noise and condition vector
        x_vec = np.random.normal(loc=0, scale=1, size=(self.tot_data_dim, num_gen))
        condition_vec = np.zeros((self.condition_dim, num_gen))
        if condition is not None: 
            for i in range(num_gen): condition_vec[condition][i] = 1
        # reshape for image history
        im = np.reshape(x_vec, (self.y_dim,self.x_dim))
        im_history.append(im)
        # sampling process
        for t in range(self.T, -1, -1): # TODO check this!
            if t > 1: z = np.random.normal(loc=0, scale=1, size=(self.tot_data_dim, num_gen))
            elif t == 1: z = np.zeros((self.tot_data_dim, num_gen))

            epsilon_pred = self._forward(activation=np.vstack((x_vec, condition_vec)))

            x_vec = ((x_vec  - ((1-self.alpha)/np.sqrt(1-(self.alpha**t)))*epsilon_pred)/np.sqrt(self.alpha)) + self.beta*z
            
            im = np.reshape(x_vec, (self.y_dim,self.x_dim))
            im_history.append(im)

        if return_history: return im_history
        return np.reshape(x_vec, (self.y_dim,self.x_dim))

    # special train function for 
    def train(self, train_data, train_conditions, batch_size, learning_rate, weight_decay, epochs, verbose=False, plot_learning=False):
        train_cost_history, valid_cost_history = [], []
        N = train_data.shape[1]
        for epoch in range(epochs):
            start = time()
            # TODO make this stochastic 
            for batch_index in range(train_data.shape[1]//batch_size):
                # TODO make for diffusion
                train_data_batch = train_data[:, range(batch_index*batch_size, ((batch_index+1)*batch_size))]
                labels_batch = train_conditions[:, range(batch_index*batch_size, ((batch_index+1)*batch_size))]

                epsilon  = np.random.normal(loc=0, scale=1, size=(self.tot_data_dim, batch_size))# NOTE noise vector epsilon
                t = np.random.randint(low=1, high=(self.T+1))

                noisy_vec = np.sqrt(self.alpha**t)*train_data_batch + np.sqrt(1-self.alpha**t)*epsilon
                # add condition vector
                noisy_vec = np.vstack((noisy_vec, labels_batch))

                train_cost = self._backward(
                    activation=noisy_vec, 
                    label=epsilon, 
                    learning_rate=learning_rate, 
                    weight_decay=weight_decay, 
                    N=N
                )
            train_cost_history.append(train_cost)
            end = time()
            
            if verbose and (epoch % 2) == 0: 
                print(f'Training cost after epoch {epoch} = {train_cost}. Completed in {end-start:.4f}s') 
        
        if plot_learning: # plot learning curves
            plt.plot([i for i in range(1, epochs+1)], train_cost_history, label=f'Train')
            plt.title(f'Training and validation cost per epoch')
            plt.legend(loc='upper right')
            plt.xlabel(f'Epoch')
            plt.ylabel(f'Cost (MSE)')
            plt.show()

    # NOTE deprecated
    def prep_data_for_diffusion(self, x, y, T):
        x = x.transpose()
        y = y.transpose()

        train_data, train_labels = [], []
        for data_vec, label_vec in zip(x, y):
            noise = np.random.normal(loc=self.loc, scale=self.scale, size=data_vec.shape)
            for t in range(T+1):
                temp_vec = ((T-t)/T)*data_vec + (t/T)*noise
                temp_vec = np.reshape(temp_vec, (self.tot_data_dim, 1)) # TODO this is worse
                label_vec = np.reshape(label_vec, (self.condition_dim, 1))
                temp_vec = np.vstack((temp_vec, label_vec))
                train_data.append(temp_vec)
                if self.pred_data: train_labels.append(np.vstack((np.reshape(data_vec, (self.tot_data_dim,1)), label_vec)))
                elif not self.pred_data: train_labels.append(np.vstack((np.reshape(noise, (self.tot_data_dim,1)), label_vec)))

        train_data = np.array(train_data).transpose()
        train_labels = np.array(train_labels).transpose()

        train_data = np.reshape(train_data, (self.tot_dim, -1))
        train_labels = np.reshape(train_labels, (self.tot_dim, -1))

        print(f'Data prepared for diffusion training.')
        return train_data, train_labels


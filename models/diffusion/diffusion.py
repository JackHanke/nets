from ..ann.ann import ArtificialNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pickle

# example training script for diffusion model
def train_diff(model, train_data, train_conditions, batch_size, epochs, optimizer, valid_data=None, valid_conditions=None, verbose=False, plot_learning=False):
    train_cost_history, valid_cost_history = [], []
    N = train_data.shape[1]
    for epoch in range(epochs):
        start = time()
        for batch_index in range(train_data.shape[1]//batch_size):
            start_batch = time()
            train_data_batch = train_data[:, range(batch_index*batch_size, ((batch_index+1)*batch_size))]
            if train_conditions is not None: labels_batch = train_conditions[:, range(batch_index*batch_size, ((batch_index+1)*batch_size))]

            t = np.random.randint(low=1, high=(model.T+1), size=(train_data_batch.shape[1],))

            noisy_vec, epsilon = model.get_x_t(data=train_data_batch, t=t)

            # make one-hot time vector
            time_vec = np.eye(model.T)[t-1].transpose()

            # add time vector (and condition vector)
            if train_conditions is None: vec = np.vstack((noisy_vec, time_vec))
            elif train_conditions is not None: vec = np.vstack((noisy_vec, labels_batch, time_vec))

            
            train_cost, _ = model._backward(
                activation=vec, 
                label=epsilon, 
                N=None
            )

            optimizer.step(
                weights = model.weights, 
                weights_gradients = model.weights_gradients,
                biases = model.biases,
                biases_gradients = model.biases_gradients
            )

            if valid_data is not None:
                noisy_vec, epsilon = model.get_x_t(data=valid_data, t=t)
                if train_conditions is None: vec = np.vstack((noisy_vec, time_vec))
                elif train_conditions is not None: vec = np.vstack((noisy_vec, labels_batch, time_vec))
                validation_inferences = model._forward(activation=vec)
                validation_cost = model.loss.cost(validation_inferences, epsilon)
                valid_cost_history.append(np.log(validation_cost))

        train_cost_history.append(train_cost) # NOTE logged
                
        end = time()
        if verbose and (epoch % 1) == 0: 
            print(f'Training cost after epoch {epoch} = {train_cost}. Completed in {end-start:.4f}s') 

        if epoch % 5 == 4:
            path_str = f'models/diffusion/saves/mnist_diffusion_{model.version_num}.pkl'
            with open(path_str, 'wb') as f:
                pickle.dump(model, file=f)
            print(f'Model saved at: {path_str}')
    
        if plot_learning: # plot learning curves
            plt.plot([i for i in range(len(train_cost_history))], train_cost_history, label=f'Train')
            if valid_data is not None: plt.plot([i for i in range(len(valid_cost_history))], valid_cost_history, label=f'Valid')
            plt.title(f'Epoch {epoch} has train cost {train_cost:.6f}')
            plt.legend(loc='upper right')
            plt.xlabel(f'Epoch')
            plt.ylabel(f'Cost (MSE)')
            plt.pause(0.000001)
        if plot_learning and epoch != epochs-1: plt.cla()
    if plot_learning:
        plt.title(f'Completed {epochs} epochs at train cost {train_cost:.6f}.')
        plt.show()


# NOTE citations
# This class was written after reading the following papers 
#   Ho et al.: https://arxiv.org/pdf/2006.11239
#   Calvin Luo: https://arxiv.org/pdf/2208.11970 
# as well as the YouTube video by Algorithmic Simplicity: https://www.youtube.com/watch?v=zc5NTeJbk-k
# https://selflein.github.io/diffusion_practical_guide
class Diffusion(ArtificialNeuralNetwork):
    def __init__(self, T, x_dim, y_dim, color_dim, condition_dim, pred_data=True, **kwargs):
        super(Diffusion, self).__init__(**kwargs)
        self.T = T # the number of diffusion steps
        self.x_dim = x_dim # dimension of horizontal pixels in image
        self.y_dim = y_dim # dimension of vertical pixels in image
        self.color_dim = color_dim # number of color channels 
        self.tot_data_dim = self.x_dim*self.y_dim*self.color_dim
        self.condition_dim = condition_dim # dimension of condition vector
        self.tot_dim = self.tot_data_dim + self.condition_dim
        self.pred_data = pred_data # boolean on whether the neural network should predict the source image or the source noise
        self.betas = np.linspace(start=1e-4, stop=2e-1, num=self.T+1) # noise schedule
        self.alphas = 1. - self.betas
        self.alpha_bars = np.cumprod(self.alphas, axis=-1)

    def alpha(self, t): return self.alphas[t]

    def alpha_bar(self, t): return self.alpha_bars[t]

    def sigma(self, t): # TODO fix for vector t!
        if t == 1: return 0
        return (1-self.alpha(t=t)) * (1-self.alpha_bar(t=t-1))/(1-self.alpha_bar(t=t))

    # takes data and applies the forward process to time vector t
    def get_x_t(self, data, t):
        epsilon = np.random.normal(loc=0, scale=1, size=data.shape)
        alpha_bar = self.alpha_bar(t=t)

        mean_term = np.sqrt(alpha_bar)
        std_term = np.sqrt(1. - alpha_bar)

        return (mean_term * data) + (std_term * epsilon), epsilon

    # takes previous vector and noise prediction and applies the backward process for scalar t
    def get_x_t_minus_1(self, x_vec, noise_pred, t):
        z = np.random.normal(loc=0, scale=1, size=x_vec.shape)
        noise_term = ((1. - self.alpha(t=t))/np.sqrt(1. - (self.alpha_bar(t=t))))
        return (x_vec  - noise_term*noise_pred)/np.sqrt(self.alpha(t=t)) +  self.sigma(t=t) * z

    # generate num_gen image(s) with a given condition(index for one-hot), 
    def gen(self, condition=None, num_gen=1, return_history=False):
        im_history = []
        # initialize noise and condition vector
        x_vec = np.random.normal(loc=0, scale=1, size=(self.tot_data_dim, num_gen))
        condition_vec = np.zeros((self.condition_dim, num_gen))
        if condition is not None: 
            for i in range(num_gen): condition_vec[condition][i] = 1
        # reshape for image history
        if self.tot_data_dim > 2: im = np.reshape(x_vec, (self.y_dim, self.x_dim))
        else: im = x_vec
        im_history.append(im)
        # sampling process
        for t in range(self.T, 0, -1): # TODO check this!
            # prepare input vector
            time_vec = np.zeros((self.T, num_gen))
            for i in range(num_gen): time_vec[t-1][i] = 1
            if self.condition_dim == 0: vec = np.vstack((x_vec, time_vec))
            elif self.condition_dim > 0: vec = np.vstack((x_vec, condition_vec, time_vec))
            # predict noise
            noise_pred = self._forward(activation=vec)
            # get result of reverse process
            x_vec = self.get_x_t_minus_1(x_vec=x_vec, noise_pred=noise_pred, t=t)
            
            # TODO make something better than this check
            # record intermediate images
            if self.tot_data_dim > 2: im = np.reshape(x_vec, (self.y_dim, self.x_dim))
            else: im = x_vec
            im_history.append(im)

        if return_history: return im_history
        return np.reshape(x_vec, (self.y_dim,self.x_dim))

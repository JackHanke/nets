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
    def __init__(self, T, x_dim, y_dim, color_dim, condition_dim, pred_data=True, **kwargs):
        super(Diffusion, self).__init__(**kwargs)
        self.T = T # the number of diffusion steps
        # self.alpha = linear_alpha # noise schedule function
        self.x_dim = x_dim # dimension of horizontal pixels in image
        self.y_dim = y_dim # dimension of vertical pixels in image
        self.color_dim = color_dim # number of color channels 
        self.tot_data_dim = self.x_dim*self.y_dim*self.color_dim
        self.condition_dim = condition_dim # dimension of condition vector
        self.tot_dim = self.tot_data_dim + self.condition_dim
        self.pred_data = pred_data # boolean on whether the neural network should predict the source image or the source noise

    # noise schedule

    # def alpha(self, t):
    #     return 0.999

    def alpha(self, t):
        high = 1-0.02
        low = 1-0.0001
        m = (high-low)/(self.T - 1) # NOTE minus 1 ?
        b = low - m
        return m*t + b

    # special train function for 
    def train(self, train_data, train_conditions, batch_size, learning_rate, weight_decay, epochs, verbose=False, plot_learning=False):
        train_cost_history = []
        N = train_data.shape[1]
        for epoch in range(epochs):
            start = time()
            # TODO make batching stochastic 
            for batch_index in range(train_data.shape[1]//batch_size):
                train_data_batch = train_data[:, range(batch_index*batch_size, ((batch_index+1)*batch_size))]
                labels_batch = train_conditions[:, range(batch_index*batch_size, ((batch_index+1)*batch_size))]

                t = np.random.randint(low=1, high=(self.T+1))

                epsilon = np.random.normal(loc=0, scale=1, size=(self.tot_data_dim, batch_size))

                alpha_bar = 1
                for t_star in range(1,t+1): alpha_bar *= self.alpha(t=t_star)

                # noisy_vec = (np.sqrt(alpha_bar)*train_data_batch) + (np.sqrt(1-alpha_bar)*epsilon)
                noisy_vec = (np.sqrt(self.alpha(t=t))*train_data_batch) + (np.sqrt(1-self.alpha(t=t))*epsilon)

                # add condition vector
                vec = np.vstack((noisy_vec, labels_batch, t*np.ones((1, batch_size))))

                if 0: # TODO remove all this nonsense
                    tl = np.reshape(noisy_vec, (28, 28))
                    tr = np.reshape(epsilon, (28, 28))
                    bl = -1*np.ones((28, 28))
                    br = np.reshape(self._forward(activation=vec), (28, 28))
                    b = np.hstack((bl, br))
                    top = np.hstack((tl, tr))
                    im = np.vstack((top, b))
                    im = plt.imshow(im, vmin=0, vmax=1)
                    plt.set_cmap('Grays')
                    plt.clim(-1, 1)
                    plt.axis('off')
                    
                    title_str = ''
                    title_str += f't={t} '
                    #  {sqrt(self.alpha**t):.1f} {sqrt(1-self.alpha**t):.1f} '
                    # title_str += f'tdb = {np.max(train_data_batch):.1f} {np.min(train_data_batch):.1f} '
                    title_str += f'epsilon = {np.max(epsilon):.1f} {np.min(epsilon):.1f} '
                    title_str += f'noisy_vec = {np.max(noisy_vec):.1f} {np.min(noisy_vec):.1f} '
                    title_str += f'max pred = {np.max(br):.1f} {np.min(br):.1f} '
                    plt.title(title_str)
                    plt.show()

                train_cost = self._backward(
                    activation=vec, 
                    label=epsilon, 
                    learning_rate=learning_rate, 
                    weight_decay=weight_decay, 
                    N=None  # NOTE something weird with regularization
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

    # generate num_gen image(s) with a given condition(index for one-hot), 
    def gen(self, condition=None, num_gen=1, return_history=False):
        im_history = []
        # initialize noise and condition vector
        x_vec = np.random.normal(loc=0, scale=1, size=(self.tot_data_dim, num_gen))
        condition_vec = np.zeros((self.condition_dim, num_gen))
        if condition is not None: 
            for i in range(num_gen): condition_vec[condition][i] = 1
        # reshape for image history
        im = np.reshape(x_vec, (self.y_dim, self.x_dim))
        im_history.append(im)
        # sampling process
        for t in range(self.T, 0, -1): # TODO check this!
            if t > 1: z = np.random.normal(loc=0, scale=1, size=(self.tot_data_dim, num_gen))
            elif t == 1: z = np.zeros((self.tot_data_dim, num_gen))

            epsilon_pred = self._forward(activation=np.vstack((x_vec, condition_vec, t*np.ones((1, num_gen)))))

            alpha_bar = 1
            for t_star in range(1,t+1): alpha_bar *= self.alpha(t=t_star)

            # x_vec = ((x_vec  - ((1-self.alpha(t=t))/np.sqrt(1-(alpha_bar)))*epsilon_pred)/np.sqrt(self.alpha(t=t))) + (1-self.alpha(t)) * z
            x_vec = ((x_vec  - np.sqrt(1-(self.alpha(t=t))))*epsilon_pred)/np.sqrt(self.alpha(t=t)) + np.sqrt(1-self.alpha(t)) * z
            
            im = np.reshape(x_vec, (self.y_dim,self.x_dim))
            im_history.append(im)

        if return_history: return im_history
        return np.reshape(x_vec, (self.y_dim,self.x_dim))

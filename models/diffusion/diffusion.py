from ..ann.ann import ArtificialNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pickle

# TODO remove this stuff
import matplotlib.animation as animation

def anim_ims(arr, save_path, fps=10, show=False):
    nSeconds = len(arr)//fps
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    a = arr[0]
    im = plt.imshow(a, vmin=0, vmax=1)
    plt.set_cmap('Grays')
    plt.clim(0,1)
    # plt.clim(-1,1)
    plt.axis('off')

    def animate_func(i):
        if i % fps == 0: print( '.', end ='' )
        im.set_array(arr[i])
        plt.title(f't={i} ({np.min(arr[i]):.2f}, {np.max(arr[i]):.2f})')
        return [im]

    anim = animation.FuncAnimation(
                fig, 
                animate_func, 
                frames = (nSeconds * fps),
                interval = (1000 / fps), # in ms
            )
    if show: plt.show()
    if not show: anim.save(save_path, fps=fps)

# animates plotted (x,y) data (2D) TODO remove
def anim_plot(arr, save_path, fps=10, show=False):
    nSeconds = len(arr)//fps
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    a = arr[0]
    im = plt.scatter(a[0], a[1])
    ax = plt.gca()
    axis_min, axis_max = -2.5, 2.5
    ax.set_xlim([axis_min, axis_max])
    ax.set_ylim([axis_min, axis_max])

    def animate_func(i):
        if i % fps == 0: print( '.', end ='' )
        # im.set_data
        plt.clf()
        ax = plt.gca()
        ax.set_xlim([axis_min, axis_max])
        ax.set_ylim([axis_min, axis_max])
        im = plt.scatter(arr[i][0], arr[i][1])
        plt.title(f't={i+1} ({np.min(arr[i]):.2f}, {np.max(arr[i]):.2f})')
        return [im]

    anim = animation.FuncAnimation(
                fig, 
                animate_func, 
                frames = (nSeconds * fps),
                interval = (1000 / fps), # in ms
            )
    print('Animation made.')
    if show: plt.show()
    if not show: anim.save(save_path, fps=fps)

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
        # self.alpha = linear_alpha # noise schedule function
        self.x_dim = x_dim # dimension of horizontal pixels in image
        self.y_dim = y_dim # dimension of vertical pixels in image
        self.color_dim = color_dim # number of color channels 
        self.tot_data_dim = self.x_dim*self.y_dim*self.color_dim
        self.condition_dim = condition_dim # dimension of condition vector
        self.tot_dim = self.tot_data_dim + self.condition_dim
        self.pred_data = pred_data # boolean on whether the neural network should predict the source image or the source noise
        # self.betas = np.linspace(start=0.0, stop=0.99, num=self.T+1)**4 # noise schedule
        self.betas = np.linspace(start=1e-4, stop=2e-1, num=self.T+1) # noise schedule
        self.alphas = 1. - self.betas
        self.alpha_bars = np.cumprod(self.alphas, axis=-1)

        # plt.plot(self.betas, label=f'betas')
        # plt.plot([1-self.alpha_bar(t) for t in range(self.T)], label=f"1 - alpha_bar")
        # plt.legend(loc=f'upper left')
        # plt.show()

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

    # special train function for 
    def train(self, train_data, train_conditions, batch_size, learning_rate, weight_decay, epochs, valid_data=None, valid_conditions=None, verbose=False, plot_learning=False):
        train_cost_history = []
        valid_cost_history = []
        N = train_data.shape[1]
        for epoch in range(epochs):
            start = time()
            # TODO make batching stochastic 
            for batch_index in range(train_data.shape[1]//batch_size):
                start_batch = time()
                train_data_batch = train_data[:, range(batch_index*batch_size, ((batch_index+1)*batch_size))]
                if train_conditions is not None: labels_batch = train_conditions[:, range(batch_index*batch_size, ((batch_index+1)*batch_size))]

                t = np.random.randint(low=1, high=(self.T+1), size=(train_data_batch.shape[1],))

                noisy_vec, epsilon = self.get_x_t(data=train_data_batch, t=t)

                if 0:
                    plt.scatter(noisy_vec[0], noisy_vec[1])
                    ax = plt.gca()
                    axis_min, axis_max = -1.5, 1.5
                    ax.set_xlim([axis_min, axis_max])
                    ax.set_ylim([axis_min, axis_max])
                    plt.title(f't={t}')
                    plt.show()

                # make one-hot time vector
                time_vec = np.eye(self.T)[t-1].transpose()

                # add time vector (and condition vector)
                if train_conditions is None: vec = np.vstack((noisy_vec, time_vec))
                elif train_conditions is not None: vec = np.vstack((noisy_vec, labels_batch, time_vec))

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

                if valid_data is not None:
                    noisy_vec, epsilon = self.get_x_t(data=valid_data, t=t)
                    if train_conditions is None: vec = np.vstack((noisy_vec, time_vec))
                    elif train_conditions is not None: vec = np.vstack((noisy_vec, labels_batch, time_vec))
                    validation_inferences = self._forward(activation=vec)
                    validation_cost = self.loss.cost(validation_inferences, epsilon)
                    valid_cost_history.append(np.log(validation_cost))

                print(f' > cost for batch {batch_index} = {train_cost} ({time()-start_batch:.4f}s)') # TODO remove this
                train_cost_history.append(np.log(train_cost)) # NOTE logged
            
                # if (batch_index % 100) == 0:
                    # vec_history = self.gen(num_gen=1000, return_history=True)
                    # anim_plot(arr=vec_history, save_path=f'models/diffusion/moon-anim.gif', fps=5, show=True)
                    
            end = time()
            if verbose and (epoch % 1) == 0: 
                print(f'Training cost after epoch {epoch} = {train_cost}. Completed in {end-start:.4f}s') 

            if epoch % 5 == 4:
                path_str = f'models/diffusion/saves/mnist_diffusion_{self.version_num}.pkl'
                with open(path_str, 'wb') as f:
                    pickle.dump(self, file=f)
                print(f'Model saved at: {path_str}')
        
        if plot_learning: # plot learning curves
            # train_cost_history = train_cost_history[500:]
            # valid_cost_history = valid_cost_history[500:]
            plt.plot([i for i in range(len(train_cost_history))], train_cost_history, label=f'Train')
            if valid_data is not None: plt.plot([i for i in range(len(valid_cost_history))], valid_cost_history, label=f'Valid')
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

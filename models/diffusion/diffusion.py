from ..ann.ann import ArtificialNeuralNetwork
import numpy as np

# NOTE
# This class was written after reading the following paper by Calvin Luo: https://arxiv.org/pdf/2208.11970 
# as well as the YouTube video by Algorithmic Simplicity: https://www.youtube.com/watch?v=zc5NTeJbk-k
class Diffusion(ArtificialNeuralNetwork):
    def __init__(self, T, x_dim, y_dim, color_dim, condition_dim, **kwargs):
        super(Diffusion, self).__init__(**kwargs)
        self.T = T # the number of diffusion steps
        self.x_dim = x_dim # dimension of horizontal pixels in image
        self.y_dim = y_dim # dimension of vertical pixels in image
        self.color_dim = color_dim # number of color channels 
        self.tot_data_dim = self.x_dim*self.y_dim*self.color_dim
        self.condition_dim = condition_dim # dimension of condition vector
        self.tot_dim = self.tot_data_dim + self.condition_dim

    # generate num_gen image(s) with a given condition(index for one-hot), 
    def gen(self, condition=None, num_gen=1, path=None, anim_path=None):
        # TODO still needs work
        im_history = []
        x_vec = np.random.normal(loc=0, scale=1, size=(self.tot_data_dim))
        condition_vec = np.zeros((self.condition_dim, num_gen))
        if condition is not None: 
            for i in range(num_gen): condition_vec[condition][i] = 1
        x_vec = np.vstack((x_vec, condition_vec))
        im_history.append(np.reshape(x_vec[:-1*self.condition_dim], (self.y_dim,self.x_dim)))
        for t in range(diff.T): 
            x_vec = self._forward(activation=x_vec)
            im_history.append(np.reshape(x_vec[:-1*self.condition_dim], (self.y_dim,self.x_dim)))
        return x_vec

    # TODO time this and make it better and faster
    def prep_data_for_diffusion(self, x, y, T):
        x = x.transpose()
        y = y.transpose()

        train_data, train_labels = [], []
        for data_vec, label_vec in zip(x, y):
            noise = np.random.normal(loc=(1/2), scale=(1/4), size=data_vec.shape)
            for t in range(T+1):
                temp_vec = ((T-t)/T)*data_vec + (t/T)*noise
                temp_vec = np.reshape(temp_vec, (self.tot_data_dim, 1)) # TODO this is worse
                label_vec = np.reshape(label_vec, (self.condition_dim, 1))
                temp_vec = np.vstack((temp_vec, label_vec))
                train_data.append(temp_vec)
                train_labels.append(np.vstack((np.reshape(data_vec, (self.tot_data_dim,1)), label_vec)))

        train_data = np.array(train_data).transpose()
        train_labels = np.array(train_labels).transpose()

        train_data = np.reshape(train_data, (self.tot_dim, -1))
        train_labels = np.reshape(train_labels, (self.tot_dim, -1))

        print(f'Data prepared for diffusion training.')
        return train_data, train_labels


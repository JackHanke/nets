from ..ann.ann import ArtificialNeuralNetwork
import numpy as np

# Constructed from: https://arxiv.org/pdf/2208.11970

class Diffusion(ArtificialNeuralNetwork):
    def __init__(self, T, x_dim, y_dim, color_dim, condition_dim, **kwargs):
        super(Diffusion, self).__init__(**kwargs)
        self.T = T # the number of diffusion steps
        self.x_dim = x_dim # dimension of horizontal pixels in image
        self.y_dim = y_dim # dimension of vertical pixels in image
        self.color_dim = color_dim # number of color channels 
        self.condition_dim = condition_dim # dimension of condition vector

    # generate num_gen image(s) with a given condition(index for one-hot), 
    def gen(self, condition=None, num_gen=1, path=None, anim_path=None):
        # TODO still needs work
        im_history = []
        x_vec = np.random.normal(loc=0, scale=1, size=(self.x_dim*self.y_dim*self.color_dim, num_gen))
        condition_vec = np.zeros((self.condition_dim, num_gen))
        if condition is not None: 
            for i in range(num_gen): condition_vec[condition][i] = 1
        x_vec = np.vstack((x_vec, condition_vec))
        im_history.append(np.reshape(x_vec[:-1*self.condition_dim], (self.y_dim,self.x_dim)))
        for t in range(diff.T): 
            x_vec = self._forward(activation=x_vec)
            im_history.append(np.reshape(x_vec[:-1*self.condition_dim], (self.y_dim,self.x_dim)))
        return x_vec

# TODO time this an make it better and faster
def prep_data_for_diffusion(x, y, T):
    x = x.transpose()
    y = y.transpose()

    train_data, train_labels = [], []
    for data_vec, label_vec in zip(x, y):
        noise = np.random.normal(loc=(1/2), scale=(1/4), size=data_vec.shape)
        for t in range(T+1):
            temp_vec = ((T-t)/T)*data_vec + (t/T)*noise
            temp_vec = np.reshape(temp_vec, (784, 1)) # TODO this is worse
            label_vec = np.reshape(label_vec, (10, 1))
            temp_vec = np.vstack((temp_vec, label_vec))
            train_data.append(temp_vec)
            train_labels.append(np.vstack((np.reshape(data_vec, (784,1)), label_vec)))

    train_data = np.array(train_data).transpose()
    train_labels = np.array(train_labels).transpose()

    train_data = np.reshape(train_data, (794, -1))
    train_labels = np.reshape(train_labels, (794, -1))

    print(f'Data prepared for diffusion training.')
    return train_data, train_labels


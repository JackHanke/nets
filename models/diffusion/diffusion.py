import numpy as np
from dataload import get_mnist

# diffusion parameter
alpha = 0.03

np.random.seed(343)

im = np.zeros((28,28))
noise = np.random.normal(loc=0, scale=1, size=(28,28))

print(im)
print(im+noise)

class Diffusion:
    def __init__(self):
        pass

from models.vae.vae import VariationalAutoEncoder
import numpy as np

vae = VariationalAutoEncoder()

x = np.random.uniform(size=(784,1))
print(x.transpose())

x_prime = vae.forward(x)

print(x_prime.transpose())

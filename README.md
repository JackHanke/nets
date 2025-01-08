# Neural Networks from Scratch
This repo hosts custom implementations of **Numpy-only** neural network architectures. Implemented and planned architectures include: 
- ✓ An artificial neural network (ANN)
    - ✓ Regularization (written as weight decay)
    - Isolate weight update and optimizer from gradient calculating process
    - ADAM optimizer
    - Dropout
    - Automatic differentiation
    - GPU-ify with [CuPy](https://cupy.dev/)
- ✓ Autoencoder
- Variational autoencoder
- Denoising diffusion model
    - Classifier free guidance
- Convolutional neural network (CNN)
- Transformer
    - Sentence-embedding model
    - Text-to-image model

## Datasets used for Benchmarking
Datasets used in this projects include:
- [MNIST](https://www.tensorflow.org/datasets/catalog/mnist)
- [Iris](https://archive.ics.uci.edu/dataset/53/iris)
- [Half Moon](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)

## Best Results Achieved

| Model Type | Arch | Regularization | Dropout | Epoch| Lr | Dataset | Acc |
|-|-|-|-|-|-|-|-|
| ANN | (784,) | L2 | 0 | 0 | 0 | MNIST | 0 |
| ANN | (4,) | L2 | 0 | 0 | 0 | Iris | 0 |

## Resources
- [Introduction to Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen

## Project TODOs
- ANN
    - Debug CrossEntropy class
- Diffusion Model
    - Conditional image generation with a diffusion model from scratch on mnist
    - For Linkedin post:
        - Generative AI from scratch! I wrote a latent denoising diffusion model with just NumPy to "draw" requested letters using the EMNIST dataset. This model architecture is used in modern image generators like Stable Diffusion and DALLE.

        - For viz, have header with "Please draw a __.", and have it draw 1738 (ay).

## Notes
For layers $2 \leq \ell \leq L$, a feed forward neural network is defined by

$$z^{\ell} = w^{\ell}a^{\ell-1} + b^{\ell}$$
$$a^{\ell} = \sigma^{\ell}(z^{\ell})$$

for activation functions $\sigma^{\ell}$ and weights $w^{\ell}$ and biases $b^{\ell}$.

>  NOTE: Superscripts are not exponentiation unless stated. 

[We define](http://neuralnetworksanddeeplearning.com/chap2.html) the error $\delta_j^{\ell}$ of neuron $j$ at layer $\ell$ be 

$$\delta_j^{\ell} = \frac{\partial C}{\partial z_j^\ell}$$ 

Then backpropagating the error can be conducted using the following equations

$$\delta^L = \nabla_a C \cdot \sigma'(z^L)$$
$$\delta^{\ell} = ((w^{\ell+1})^{T}\delta^{\ell+1}) \cdot \sigma'(z^{\ell})$$
$$\frac{\partial C}{\partial b_j^\ell} = \delta_j^{\ell}$$
$$\frac{\partial C}{\partial w_{jk}^\ell} = a_k^{\ell-1}\delta_j^{\ell}$$

For VAE's that consist of encoder $E$ and decoder $D$, we have the following loss function

$$\cal{L} = \cal{L}_{rec} + \cal{L}_{rec} = \frac{1}{2}\sum_{i}(x_i - x_i')^2 - \frac{1}{2}\sum_{i}(1+2\log(\sigma_i)-\mu_i^2-\sigma_i^2).$$

$$\cal{L}_{reg} = \frac{1}{2}\exp(2\log\sigma) + \frac{1}{2}\mu^2 - \log\sigma - \frac{1}{2}$$

If we let $z$ be the input to

$$z = \mu + \epsilon \exp(\log \sigma)$$


$\frac{\partial \cal{L}}{\partial z}$ is computable after a VAE forward pass and a $D$ backprop. 

Let 

$$\frac{\partial \cal{L}_{rec}}{\partial \mu} = \frac{\partial \cal{L}_{rec}}{\partial z}\frac{\partial z}{\partial \mu} = \frac{\partial \cal{L}_{rec}}{\partial z}$$

$$\frac{\partial \cal{L}_{rec}}{\partial \log{\sigma}} = \frac{\partial \cal{L}_{rec}}{\partial z}\frac{\partial z}{\partial \log{\sigma}} = \frac{\partial \cal{L}_{rec}}{\partial z} \epsilon \exp(\log\sigma)$$

$$\frac{\partial \cal{L}_{reg}}{\partial \mu} = \mu$$

$$\frac{\partial \cal{L}_{reg}}{\partial \log{\sigma}} = \exp(2\log \sigma) - \vec{1}$$




# Calculations for Convolutional Neural Network

## Usual Backpropagation

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

## Usual Backpropagation

For a CNN layer TODO

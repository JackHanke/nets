# Variational Autoencoders

## VAE Infernece

$$(\mu, \sigma) = q_{\phi}(z|x)$$

$$\epsilon \sim \cal{N}(0,I)$$

$$z = \mu + \sigma \circ \epsilon$$

$$x' = p_{\theta}(x|z)$$

## VAE Objective

The VAE training objective seeks to minimize

$$\cal{L} = \cal{L}_2(x,x') -\frac{1}{2}(1+\log(\sigma^2) - \mu^2 - \sigma^2)$$

for $p_\theta(z) = \cal{N}(z;0,I)$ and $q_{\phi}(z|x) = \cal{N}(z;\mu,\sigma^2 I)$, where 

$$\cal{L}_2(x,x') = \frac{1}{2}\sum_{i}(x_i - x'_i)^2.$$

## VAE Gradients

$\mu, \sigma$ are just functions of $\phi$, the gradient 

$$\frac{\partial \cal{L}}{\partial \theta} = $$

But what is

$$\frac{\partial \cal{L}}{\partial \phi} = ?$$

## TODOs

1. Question: Is this $\cal{L}_2$ or $\cal{L}^2_2$?

2. Show 
$$D_{KL}(\cal{N}(\mu,\sigma^2 I) || \cal{N}(0,I)) = -\frac{1}{2}\sum_{i}(1+\log(\sigma_i^2) - \mu_i^2 - \sigma_i^2)$$

where $$D_{KL}(P || Q) = \sum_{x \in \cal{X}}P(x)\log \left(\frac{P(x)}{Q(x)}\right).$$

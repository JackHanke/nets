# Neural Networks from Scratch
This repo consists of academic exercises to create all facets of the following neural network architectures from scratch.
- A multilayer perceptron, including
    - Regularization
    - Dropout
- A convolutional neural network

Here scratch means just using Python and NumPy. 

Implementations will be benchmarked on both the classical [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) and [Iris](https://archive.ics.uci.edu/dataset/53/iris) data sets.

## Best Results Achieved

| Model Type | Arch | Regularization | Dropout | Epoch| Lr | Dataset | Acc |
|-|-|-|-|-|-|-|-|
| ANN | (784,) | L2 | 0 | 0 | 0 | MNIST | 0 |
| ANN | (4,) | L2 | 0 | 0 | 0 | Iris | 0 |

## Resources
- [Introduction to Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen

## Project TODOs
- Implement model save and load feature
- Debug CrossEntropy class
- Add dropout
- Implement Adam Optimizer
- Add CNN
- Diffusion!
    - Following [this](https://www.youtube.com/watch?v=zc5NTeJbk-k&t=29s) video
    - Conditional image generation with a diffusion model from scratch on mnist
        - Train with classifier free guidance for better performance
    - For linkedin post:
        - Generative AI from scratch! I made a diffusion model using classifier free guidance to do conditional image generation, in this case drawing handwritten digits for a given digit. Diffusion models are a part of modern image generators like Stable Diffusion and DALLE. Diffusion models work by learning to predict added noise to an image TODO
        - For viz, have header with "Please draw a __.", and have it draw 1738 (ay).






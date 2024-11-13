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
- ANN
    - Implement model save and load feature
    - Debug CrossEntropy class
    - Add dropout
    - Implement Adam Optimizer
- CNN
    - Everything
- Diffusion Model
    - Follow [this](https://www.youtube.com/watch?v=zc5NTeJbk-k&t=29s) video
    - Conditional image generation with a diffusion model from scratch on mnist
        - Train with classifier free guidance for better performance
    - For Linkedin post:
        - I wrote an image generator from scratch! This program learns to draw numbers after looking at pictures of other drawings of numbers (LeCun's MNIST). 
        
        This is specifically a diffusion model using classifier free guidance, which learns to predict a given image's true pixels after noise is added. The learned network is then used to predict data of a given label and noise array, effectively generating new images. This is one of the techniques used in modern image generators like Stable Diffusion and DALLE.

        - For viz, have header with "Please draw a __.", and have it draw 1738 (ay).






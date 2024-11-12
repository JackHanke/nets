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
- differentiate between train mnist and benchmark mnist, same for iris
- Debug CrossEntropy class
- Add dropout
- Implement Adam Optimizer
- Add CNN
- Diffusion!
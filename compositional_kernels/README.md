Compositional kernels
====
This package contains code that is related to the duality between neural networks and compositional kernels as developed in:
https://arxiv.org/abs/1602.05897
https://arxiv.org/abs/1702.08503
https://arxiv.org/abs/1703.07872

In a nutshell, it is based on a notion named "computational skeleton". Each skeleton gives rise to a family of neural networks as well as a reproducing kernel and a corresponding random featues scheme.
The current package implements several concepts from the above papers:
1. Given a skeleton, it generates the corresponding networks and trains them
1. Given a skeleton, it computes the associated kernel
1. Given a skeleton, it generates random features, and trains a linear classifier with respect to those features


## Installation
The following tools must be present:
1. Bazel from https://bazel.build.
1. Tensorflow from tensorflow.org.
1. Protocol buffers from https://developers.google.com/protocol-buffers/.

## MNIST example
We next explain how to run networks and kernels corresponding to a simple skeleton specified in mnist/mnist.pb.txt.

In order to build and train a neural network from the skeleton:
1. change directory to the directory containing this README.md
1. bazel build mnist/nn_trainer
1. bazel-bin/mnist/nn_trainer

In order to generate random features from the skeleton and train a linear classifier on top of them:
1. change directory to the directory containing this README.md
1. bazel build mnist/rf_trainer
1. bazel-bin/mnist/rf_trainer

The file base/config.py declares flags to specify:
1. Learning parameters such as learning rate, optimization algorithm, number of random features, and so on
1. Relevant files such as the skeleton file, where to save the model, and so on

## Contact
vineet@google.com
amitdaniely@google.com
singer@google.com
frostig@google.com

## Maintainance
@vineet-gupta

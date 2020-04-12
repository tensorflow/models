![No Maintenance Intended](https://img.shields.io/badge/No%20Maintenance%20Intended-%E2%9C%95-red.svg)
![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

# Learned Optimizer

Code for [Learned Optimizers that Scale and Generalize](https://arxiv.org/abs/1703.04813).

## Requirements

* Bazel ([install](https://bazel.build/versions/master/docs/install.html))
* TensorFlow >= v1.3

## Training a Learned Optimizer

## Code Overview
In the top-level directory, ```metaopt.py``` contains the code to train and test a learned optimizer. ```metarun.py``` packages the actual training procedure into a
single file, defining and exposing many flags to tune the procedure, from selecting the optimizer type and problem set to more fine-grained hyperparameter settings.
There is no testing binary; testing can be done ad-hoc via ```metaopt.test_optimizer``` by passing an optimizer object and a directory with a checkpoint.

The ```optimizer``` directory contains a base ```trainable_optimizer.py``` class and a number of extensions, including the ```hierarchical_rnn``` optimizer used in
the paper, a ```coordinatewise_rnn``` optimizer that more closely matches previous work, and a number of simpler optimizers to demonstrate the basic mechanics of
a learnable optimizer.

The ```problems``` directory contains the code to build the problems that were used in the meta-training set.

### Binaries
```metarun.py```: meta-training of a learned optimizer

### Command-Line Flags
The flags most relevant to meta-training are defined in ```metarun.py```. The default values will meta-train a HierarchicalRNN optimizer with the hyperparameter
settings used in the paper.

### Using a Learned Optimizer as a Black Box
The ```trainable_optimizer``` inherits from ```tf.train.Optimizer```, so a properly instantiated version can be used to train any model in any APIs that accept
this class. There are just 2 caveats:

1. If using the Hierarchical RNN optimizer, the apply_gradients return type must be changed (see comments inline for what exactly must be removed)

2. Care must be taken to restore the variables from the optimizer without overriding them. Optimizer variables should be loaded manually using a pretrained checkpoint
and a ```tf.train.Saver``` with only the optimizer variables. Then, when constructing the session, ensure that any automatic variable initialization does not
re-initialize the loaded optimizer variables.

## Contact for Issues

* Olga Wichrowska (@olganw), Niru Maheswaranathan (@nirum)

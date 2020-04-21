![No Maintenance Intended](https://img.shields.io/badge/No%20Maintenance%20Intended-%E2%9C%95-red.svg)
![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

# Learning to Protect Communications with Adversarial Neural Cryptography

This is a slightly-updated model used for the paper
["Learning to Protect Communications with Adversarial Neural
Cryptography"](https://arxiv.org/abs/1610.06918).

> We ask whether neural networks can learn to use secret keys to protect
> information from other neural networks. Specifically, we focus on ensuring
> confidentiality properties in a multiagent system, and we specify those
> properties in terms of an adversary. Thus, a system may consist of neural
> networks named Alice and Bob, and we aim to limit what a third neural
> network named Eve learns from eavesdropping on the communication between
> Alice and Bob. We do not prescribe specific cryptographic algorithms to
> these neural networks; instead, we train end-to-end, adversarially.
> We demonstrate that the neural networks can learn how to perform forms of
> encryption and decryption, and also how to apply these operations
> selectively in order to meet confidentiality goals.

This code allows you to train encoder/decoder/adversary network triplets
and evaluate their effectiveness on randomly generated input and key
pairs.

## Prerequisites

The only software requirements for running the encoder and decoder is having
TensorFlow installed.

Requires TensorFlow r0.12 or later.

## Training and evaluating

After installing TensorFlow and ensuring that your paths are configured
appropriately:

```
python train_eval.py
```

This will begin training a fresh model.  If and when the model becomes
sufficiently well-trained, it will reset the Eve model multiple times
and retrain it from scratch, outputting the accuracy thus obtained
in each run.

## Model differences from the paper

The model has been simplified slightly from the one described in
the paper - the convolutional layer width was reduced by a factor
of two.  In the version in the paper, there was a nonlinear unit
after the fully-connected layer;  that nonlinear has been removed
here.  These changes improve the robustness of training.  The
initializer for the convolution layers has switched to the
`tf.contrib.layers default` of `xavier_initializer` instead of
a simpler `truncated_normal`.

## Contact information

This model repository is maintained by David G. Andersen
([dave-andersen](https://github.com/dave-andersen)).

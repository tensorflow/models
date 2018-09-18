# MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks

[TOC]

## What is MorphNet?

MorphNet is a method for learning deep network structure during training. The
key principle is continuous relaxation of the network-structure learning
problem. Specifically, we use regularizers that induce sparsity in the space of
activations of the network. The regularizers can be tailored to target the
consumption of specific resources by the network, such as FLOPs or model size.
When such a regularizer is added to the training loss and their sum is
minimized via stochastic gradient descent or a similar optimizer, the learning
problem becomes also a constrained optimization of the structure of the network,
under the constraint represented by the regularizer. The method is described in
detail in the [this paper](https://arxiv.org/abs/1711.06798), to appear in [CVPR
2018](http://cvpr2018.thecvf.com/).

## Adding a MorphNet regularizer to your training code

Your interaction with the MorphNet codebase will most likely be through
subclasses of `NetworkRegularizer`. Each subclass represents a resource that we
wish to target/constrain when optimizing the network. The MorphNet package
provides several `NetworkRegularizer`s in the `network_regularizers` directory,
as well as a framework for writing your own. The framework is described in
detail [here](g3doc/regularizers_framework.md). The interface of
`NetworkRegularizer` is given
[here](g3doc/regularizers_framework.md?#network-regularizers).

To apply a `NetworkRegularizer` to your network, your code would look similar to
the example below. The example refers to a specific type of `NetworkRegularizer`
that targets FLOPs, and to make the discussion simpler we henceforth restrict it
to this case, but generalization to an arbitrary constrained resource and an
arbitrary regularization method that targets that resource is straightforward.

```python
my_gamma_threshold = 1e-3
regularizer_strength = 1e-9
network_reg = network_regularizers.GammaFlopsRegularizer(
    [my_network_output.op], my_gamma_threshold)
my_training_loss += regularizer_strength * network_reg.get_regularization_term()
tf.summary.scalar('FLOPs', network_reg.get_cost()
```

Once you start your training, your TensorBoard will display the effective FLOP
count of the model. "Effective" is the sense that as activations are zeroed out
by the regularizer, their impact on the FLOP count is discounted.

![TensorBoardDisplayOfFlops](g3doc/tensorboard.png "Example of the TensorBoard
display of the resource regularized by MorphNet.")

The larger the `regularizer_strength`, the smaller the effective FLOP count to
which the network will converge. If `regularizer_strength` is large enough, the
FLOP count will collapse to zero, whereas if it is small enough, the FLOP count
will remain at its initial value and the network structure will not vary.
`regularizer_strength` is your knob to control where you want to be on the
price-performance curve. The `my_gamma_threshold` parameter is used for
determining when an activation is alive. It is described in more detail
[here](framework/README.md?#the-opregularizer-interface), including
an explanation for how to tune it.

## Extracting the architecture learned by MorphNet

One way to extract the structure is querying the `network_reg` object created
above. To query which activations in a given op were kept alive (as opposed to
removed) by MorphNet, your code would look similar to

```python
alive = sess.run(network_reg.opreg_manager.get_regularizer(op).alive_vector)
```

where `op` is the tensorflow op in question, and `sess` is a tf.Session object.
The result is a vector of booleans, designating which activations were kept
alive (more details can be found
[here](framework/README.md?#the-opregularizer-interface)). Typically
one would be interested in the number of alive activations, which can be
obtained by counting the `True` values in `alive`. Looping over all convolutions
and / or fully connected layers (as `op`) is typically sufficient to extract the
full structure learned by MorphNet.

## Maintainers

* Elad Eban
* Ariel Gordon, github: [gariel-google](https://github.com/gariel-google).

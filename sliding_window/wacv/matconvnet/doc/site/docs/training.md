## Using MatConvNet to train convnets

MatConvNet can be used to train models, typically by using a form of
stochastic gradient descent (SGD) and back-propagation.

The following learning demonstrators are provided in the MatConvNet
package:

- **MNIST**. See `examples/mnist/cnn_mnist.m`.
- **CIFAR**. See `examples/cifar/cnn_cifar.m`.
- **ImageNet**. See `examples/imagenet/cnn_imagenet.m`.

These demos are self-contained; MNIST and CIFAR, in particular,
automatically download and unpack the required data, so that they
should work out-of-the-box.

While MNIST and CIFAR are small datasets (by today's standard) and
training is feasible on a CPU, ImageNet requires a powerful GPU to
complete in a reasonable time (a few days!). It also requires the
`vl_imreadjpeg()` command in the toolbox to be compiled in order to
accelerate reading large batches of JPEG images and avoid starving the
GPU.

All these demos use the `example/cnn_train.m` and
`example/cnn_train_dag.m` SGD drivers, which are simple
implementations of the standard SGD with momentum, done directly in
MATLAB code. However, it should be easy to implement your own
specialized or improved solver.

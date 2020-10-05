# Orbit

Orbit is a flexible, lightweight library designed to make it easy to write
[custom training loops][custom_training] in TensorFlow 2. Orbit handles common
model training tasks such as saving checkpoints, running model evaluations, and
setting up summary writing, while giving users full control over implementing
the inner training loop. It integrates with `tf.distribute` seamlessly and
supports running on different device types (CPU, GPU, and TPU). The core code is
intended to be easy to read and fork.

See our [g3doc](g3doc) at go/orbit-trainer for additional documentation.

[custom_training]: https://www.tensorflow.org/tutorials/distribute/custom_training

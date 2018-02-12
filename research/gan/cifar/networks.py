# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Networks for GAN CIFAR example using TFGAN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from slim.nets import dcgan

tfgan = tf.contrib.gan


def _last_conv_layer(end_points):
  """"Returns the last convolutional layer from an endpoints dictionary."""
  conv_list = [k if k[:4] == 'conv' else None for k in end_points.keys()]
  conv_list.sort()
  return end_points[conv_list[-1]]


def generator(noise, is_training=True):
  """Generator to produce CIFAR images.

  Args:
    noise: A 2D Tensor of shape [batch size, noise dim]. Since this example
      does not use conditioning, this Tensor represents a noise vector of some
      kind that will be reshaped by the generator into CIFAR examples.
    is_training: If `True`, batch norm uses batch statistics. If `False`, batch
      norm uses the exponential moving average collected from population
      statistics.

  Returns:
    A single Tensor with a batch of generated CIFAR images.
  """
  images, _ = dcgan.generator(noise, is_training=is_training)

  # Make sure output lies between [-1, 1].
  return tf.tanh(images)


def conditional_generator(inputs, is_training=True):
  """Generator to produce CIFAR images.

  Args:
    inputs: A 2-tuple of Tensors (noise, one_hot_labels) and creates a
      conditional generator.
    is_training: If `True`, batch norm uses batch statistics. If `False`, batch
      norm uses the exponential moving average collected from population
      statistics.

  Returns:
    A single Tensor with a batch of generated CIFAR images.
  """
  noise, one_hot_labels = inputs
  noise = tfgan.features.condition_tensor_from_onehot(noise, one_hot_labels)

  images, _ = dcgan.generator(noise, is_training=is_training)

  # Make sure output lies between [-1, 1].
  return tf.tanh(images)


def discriminator(img, unused_conditioning, is_training=True):
  """Discriminator for CIFAR images.

  Args:
    img: A Tensor of shape [batch size, width, height, channels], that can be
      either real or generated. It is the discriminator's goal to distinguish
      between the two.
    unused_conditioning: The TFGAN API can help with conditional GANs, which
      would require extra `condition` information to both the generator and the
      discriminator. Since this example is not conditional, we do not use this
      argument.
    is_training: If `True`, batch norm uses batch statistics. If `False`, batch
      norm uses the exponential moving average collected from population
      statistics.

  Returns:
    A 1D Tensor of shape [batch size] representing the confidence that the
    images are real. The output can lie in [-inf, inf], with positive values
    indicating high confidence that the images are real.
  """
  logits, _ = dcgan.discriminator(img, is_training=is_training)
  return logits


# (joelshor): This discriminator creates variables that aren't used, and
# causes logging warnings. Improve `dcgan` nets to accept a target end layer,
# so extraneous variables aren't created.
def conditional_discriminator(img, conditioning, is_training=True):
  """Discriminator for CIFAR images.

  Args:
    img: A Tensor of shape [batch size, width, height, channels], that can be
      either real or generated. It is the discriminator's goal to distinguish
      between the two.
    conditioning: A 2-tuple of Tensors representing (noise, one_hot_labels).
    is_training: If `True`, batch norm uses batch statistics. If `False`, batch
      norm uses the exponential moving average collected from population
      statistics.

  Returns:
    A 1D Tensor of shape [batch size] representing the confidence that the
    images are real. The output can lie in [-inf, inf], with positive values
    indicating high confidence that the images are real.
  """
  logits, end_points = dcgan.discriminator(img, is_training=is_training)

  # Condition the last convolution layer.
  _, one_hot_labels = conditioning
  net = _last_conv_layer(end_points)
  net = tfgan.features.condition_tensor_from_onehot(
      tf.contrib.layers.flatten(net), one_hot_labels)
  logits = tf.contrib.layers.linear(net, 1)

  return logits

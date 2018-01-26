# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Convolutional Model class.

Uses only convolutional and fully connected layers for the inference ops.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models import model
from models.layers import variables


class ConvModel(model.Model):
  """A baseline multi GPU Model without capsule layers.

  The inference graph includes ReLU convolution layers and fully connected
  layers. The last layer is linear and has 10 units.
  """

  def _add_convs(self, input_tensor, channels):
    """Adds the convolution layers.

    Adds a series of convolution layers with ReLU nonlinearity and pooling
    after each of them.

    Args:
      input_tensor: a 4D float tensor as the input to the first convolution.
      channels: A list of channel sizes for input_tensor and following
        convolution layers. Number of channels in input tensor should be
        equal to channels[0].
    Returns:
      A 4D tensor as the output of the last pooling layer.
    """
    for i in xrange(1, len(channels)):
      with tf.variable_scope('conv{}'.format(i)) as scope:
        kernel = variables.weight_variable(
            shape=[5, 5, channels[i - 1], channels[i]], stddev=5e-2,
            verbose=self._hparams.verbose
        )
        conv = tf.nn.conv2d(
            input_tensor,
            kernel, [1, 1, 1, 1],
            padding=self._hparams.padding,
            data_format='NCHW')
        biases = variables.bias_variable([channels[i]],
                                         verbose=self._hparams.verbose)
        pre_activation = tf.nn.bias_add(conv, biases, data_format='NCHW')
        relu = tf.nn.relu(pre_activation, name=scope.name)
        if self._hparams.verbose:
          tf.summary.histogram('activation', relu)
        input_tensor = tf.contrib.layers.max_pool2d(
            relu, kernel_size=2, stride=2, data_format='NCHW', padding='SAME')

    return input_tensor

  def inference(self, features):
    """Adds the inference graph ops.

    Builds the architecture of the neural net to drive logits from features.
    The inference graph includes a series of convolution and fully connected
    layers and outputs a [batch, 10] tensor as the logits.

    Args:
      features: Dictionary of batched feature tensors like images and labels.
    Returns:
      A model.Inferred named tuple of expected outputs of the model like
      'logits' and 'remakes' for the reconstructions (to be added).
    """
    image = features['images']
    image_dim = features['height']
    image_depth = features['depth']
    image_4d = tf.reshape(image, [-1, image_depth, image_dim, image_dim])
    conv = self._add_convs(image_4d, [image_depth, 512, 256])
    hidden1 = tf.contrib.layers.flatten(conv)

    with tf.variable_scope('fc1') as scope:
      dim = hidden1.get_shape()[1].value
      weights = variables.weight_variable(shape=[dim, 1024], stddev=0.1,
                                          verbose=self._hparams.verbose)
      biases = variables.bias_variable(shape=[1024],
                                       verbose=self._hparams.verbose)
      pre_activation = tf.matmul(hidden1, weights) + biases
      hidden2 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope('softmax_layer') as scope:
      weights = variables.weight_variable(
          shape=[1024, features['num_classes']], stddev=0.1,
          verbose=self._hparams.verbose
      )
      biases = variables.bias_variable(shape=[features['num_classes']],
                                       verbose=self._hparams.verbose)
      logits = tf.matmul(hidden2, weights) + biases

    return model.Inferred(logits, None)

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

"""Training/Pretraining encoder as used in PTN (NIPS16)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def _preprocess(images):
  return images * 2 - 1


def model(images, params, is_training):
  """Model encoding the images into view-invariant embedding."""
  del is_training  # Unused
  image_size = images.get_shape().as_list()[1]
  f_dim = params.f_dim
  fc_dim = params.fc_dim
  z_dim = params.z_dim
  outputs = dict()

  images = _preprocess(images)
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1)):
    h0 = slim.conv2d(images, f_dim, [5, 5], stride=2, activation_fn=tf.nn.relu)
    h1 = slim.conv2d(h0, f_dim * 2, [5, 5], stride=2, activation_fn=tf.nn.relu)
    h2 = slim.conv2d(h1, f_dim * 4, [5, 5], stride=2, activation_fn=tf.nn.relu)
    # Reshape layer
    s8 = image_size // 8
    h2 = tf.reshape(h2, [-1, s8 * s8 * f_dim * 4])
    h3 = slim.fully_connected(h2, fc_dim, activation_fn=tf.nn.relu)
    h4 = slim.fully_connected(h3, fc_dim, activation_fn=tf.nn.relu)

    outputs['ids'] = slim.fully_connected(h4, z_dim, activation_fn=tf.nn.relu)
    outputs['poses'] = slim.fully_connected(h4, z_dim, activation_fn=tf.nn.relu)
  return outputs

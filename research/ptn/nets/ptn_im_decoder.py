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

"""Image/Mask decoder used while pretraining the network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

_FEATURE_MAP_SIZE = 8


def _postprocess_im(images):
  """Performs post-processing for the images returned from conv net.

  Transforms the value from [-1, 1] to [0, 1].
  """
  return (images + 1) * 0.5


def model(identities, poses, params, is_training):
  """Decoder model to get image and mask from latent embedding."""
  del is_training
  f_dim = params.f_dim
  fc_dim = params.fc_dim

  outputs = dict()

  with slim.arg_scope(
      [slim.fully_connected, slim.conv2d_transpose],
      weights_initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1)):
    # Concatenate the identity and pose units
    h0 = tf.concat([identities, poses], 1)
    h0 = slim.fully_connected(h0, fc_dim, activation_fn=tf.nn.relu)
    h1 = slim.fully_connected(h0, fc_dim, activation_fn=tf.nn.relu)

    # Mask decoder
    dec_m0 = slim.fully_connected(
        h1, (_FEATURE_MAP_SIZE**2) * f_dim * 2, activation_fn=tf.nn.relu)
    dec_m0 = tf.reshape(
        dec_m0, [-1, _FEATURE_MAP_SIZE, _FEATURE_MAP_SIZE, f_dim * 2])

    dec_m1 = slim.conv2d_transpose(
        dec_m0, f_dim, [5, 5], stride=2, activation_fn=tf.nn.relu)
    dec_m2 = slim.conv2d_transpose(
        dec_m1, int(f_dim / 2), [5, 5], stride=2, activation_fn=tf.nn.relu)
    dec_m3 = slim.conv2d_transpose(
        dec_m2, 1, [5, 5], stride=2, activation_fn=tf.nn.sigmoid)

    # Image decoder
    dec_i0 = slim.fully_connected(
        h1, (_FEATURE_MAP_SIZE**2) * f_dim * 4, activation_fn=tf.nn.relu)
    dec_i0 = tf.reshape(
        dec_i0, [-1, _FEATURE_MAP_SIZE, _FEATURE_MAP_SIZE, f_dim * 4])

    dec_i1 = slim.conv2d_transpose(
        dec_i0, f_dim * 2, [5, 5], stride=2, activation_fn=tf.nn.relu)
    dec_i2 = slim.conv2d_transpose(
        dec_i1, f_dim * 2, [5, 5], stride=2, activation_fn=tf.nn.relu)
    dec_i3 = slim.conv2d_transpose(
        dec_i2, 3, [5, 5], stride=2, activation_fn=tf.nn.tanh)

    outputs = dict()
    outputs['images'] = _postprocess_im(dec_i3)
    outputs['masks'] = dec_m3
  return outputs

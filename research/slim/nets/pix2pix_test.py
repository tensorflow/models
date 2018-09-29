# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Tests for pix2pix."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import pix2pix


class GeneratorTest(tf.test.TestCase):

  def _reduced_default_blocks(self):
    """Returns the default blocks, scaled down to make test run faster."""
    return [pix2pix.Block(b.num_filters // 32, b.decoder_keep_prob)
            for b in pix2pix._default_generator_blocks()]

  def test_output_size_nn_upsample_conv(self):
    batch_size = 2
    height, width = 256, 256
    num_outputs = 4

    images = tf.ones((batch_size, height, width, 3))
    with tf.contrib.framework.arg_scope(pix2pix.pix2pix_arg_scope()):
      logits, _ = pix2pix.pix2pix_generator(
          images, num_outputs, blocks=self._reduced_default_blocks(),
          upsample_method='nn_upsample_conv')

    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      np_outputs = session.run(logits)
      self.assertListEqual([batch_size, height, width, num_outputs],
                           list(np_outputs.shape))

  def test_output_size_conv2d_transpose(self):
    batch_size = 2
    height, width = 256, 256
    num_outputs = 4

    images = tf.ones((batch_size, height, width, 3))
    with tf.contrib.framework.arg_scope(pix2pix.pix2pix_arg_scope()):
      logits, _ = pix2pix.pix2pix_generator(
          images, num_outputs, blocks=self._reduced_default_blocks(),
          upsample_method='conv2d_transpose')

    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      np_outputs = session.run(logits)
      self.assertListEqual([batch_size, height, width, num_outputs],
                           list(np_outputs.shape))

  def test_block_number_dictates_number_of_layers(self):
    batch_size = 2
    height, width = 256, 256
    num_outputs = 4

    images = tf.ones((batch_size, height, width, 3))
    blocks = [
        pix2pix.Block(64, 0.5),
        pix2pix.Block(128, 0),
    ]
    with tf.contrib.framework.arg_scope(pix2pix.pix2pix_arg_scope()):
      _, end_points = pix2pix.pix2pix_generator(
          images, num_outputs, blocks)

    num_encoder_layers = 0
    num_decoder_layers = 0
    for end_point in end_points:
      if end_point.startswith('encoder'):
        num_encoder_layers += 1
      elif end_point.startswith('decoder'):
        num_decoder_layers += 1

    self.assertEqual(num_encoder_layers, len(blocks))
    self.assertEqual(num_decoder_layers, len(blocks))


class DiscriminatorTest(tf.test.TestCase):

  def _layer_output_size(self, input_size, kernel_size=4, stride=2, pad=2):
    return (input_size + pad * 2 - kernel_size) // stride + 1

  def test_four_layers(self):
    batch_size = 2
    input_size = 256

    output_size = self._layer_output_size(input_size)
    output_size = self._layer_output_size(output_size)
    output_size = self._layer_output_size(output_size)
    output_size = self._layer_output_size(output_size, stride=1)
    output_size = self._layer_output_size(output_size, stride=1)

    images = tf.ones((batch_size, input_size, input_size, 3))
    with tf.contrib.framework.arg_scope(pix2pix.pix2pix_arg_scope()):
      logits, end_points = pix2pix.pix2pix_discriminator(
          images, num_filters=[64, 128, 256, 512])
    self.assertListEqual([batch_size, output_size, output_size, 1],
                         logits.shape.as_list())
    self.assertListEqual([batch_size, output_size, output_size, 1],
                         end_points['predictions'].shape.as_list())

  def test_four_layers_no_padding(self):
    batch_size = 2
    input_size = 256

    output_size = self._layer_output_size(input_size, pad=0)
    output_size = self._layer_output_size(output_size, pad=0)
    output_size = self._layer_output_size(output_size, pad=0)
    output_size = self._layer_output_size(output_size, stride=1, pad=0)
    output_size = self._layer_output_size(output_size, stride=1, pad=0)

    images = tf.ones((batch_size, input_size, input_size, 3))
    with tf.contrib.framework.arg_scope(pix2pix.pix2pix_arg_scope()):
      logits, end_points = pix2pix.pix2pix_discriminator(
          images, num_filters=[64, 128, 256, 512], padding=0)
    self.assertListEqual([batch_size, output_size, output_size, 1],
                         logits.shape.as_list())
    self.assertListEqual([batch_size, output_size, output_size, 1],
                         end_points['predictions'].shape.as_list())

  def test_four_layers_wrog_paddig(self):
    batch_size = 2
    input_size = 256

    images = tf.ones((batch_size, input_size, input_size, 3))
    with tf.contrib.framework.arg_scope(pix2pix.pix2pix_arg_scope()):
      with self.assertRaises(TypeError):
        pix2pix.pix2pix_discriminator(
            images, num_filters=[64, 128, 256, 512], padding=1.5)

  def test_four_layers_negative_padding(self):
    batch_size = 2
    input_size = 256

    images = tf.ones((batch_size, input_size, input_size, 3))
    with tf.contrib.framework.arg_scope(pix2pix.pix2pix_arg_scope()):
      with self.assertRaises(ValueError):
        pix2pix.pix2pix_discriminator(
            images, num_filters=[64, 128, 256, 512], padding=-1)

if __name__ == '__main__':
  tf.test.main()

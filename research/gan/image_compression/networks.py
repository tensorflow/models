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
"""Networks for GAN compression example using TFGAN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from slim.nets import dcgan
from slim.nets import pix2pix


def _last_conv_layer(end_points):
  """"Returns the last convolutional layer from an endpoints dictionary."""
  conv_list = [k if k[:4] == 'conv' else None for k in end_points.keys()]
  conv_list.sort()
  return end_points[conv_list[-1]]


def _encoder(img_batch, is_training=True, bits=64, depth=64):
  """Maps images to internal representation.

  Args:
    img_batch: Stuff
    is_training: Stuff
    bits: Number of bits per patch.
    depth: Stuff

  Returns:
    Real-valued 2D Tensor of size [batch_size, bits].
  """
  _, end_points = dcgan.discriminator(
      img_batch, depth=depth, is_training=is_training, scope='Encoder')

  # (joelshor): Make the DCGAN convolutional layer that converts to logits
  # not trainable, since it doesn't affect the encoder output.

  # Get the pre-logit layer, which is the last conv.
  net = _last_conv_layer(end_points)

  # Transform the features to the proper number of bits.
  with tf.variable_scope('EncoderTransformer'):
    encoded = tf.contrib.layers.conv2d(net, bits, kernel_size=1, stride=1,
                                       padding='VALID', normalizer_fn=None,
                                       activation_fn=None)
  encoded = tf.squeeze(encoded, [1, 2])
  encoded.shape.assert_has_rank(2)

  # Map encoded to the range [-1, 1].
  return tf.nn.softsign(encoded)


def _binarizer(prebinary_codes, is_training):
  """Binarize compression logits.

  During training, add noise, as in https://arxiv.org/pdf/1611.01704.pdf. During
  eval, map [-1, 1] -> {-1, 1}.

  Args:
    prebinary_codes: Floating-point tensors corresponding to pre-binary codes.
      Shape is [batch, code_length].
    is_training: A python bool. If True, add noise. If false, binarize.

  Returns:
    Binarized codes. Shape is [batch, code_length].

  Raises:
    ValueError: If the shape of `prebinary_codes` isn't static.
  """
  if is_training:
    # In order to train codes that can be binarized during eval, we add noise as
    # in https://arxiv.org/pdf/1611.01704.pdf. Another option is to use a
    # stochastic node, as in https://arxiv.org/abs/1608.05148.
    noise = tf.random_uniform(
        prebinary_codes.shape,
        minval=-1.0,
        maxval=1.0)
    return prebinary_codes + noise
  else:
    return tf.sign(prebinary_codes)


def _decoder(codes, final_size, is_training, depth=64):
  """Compression decoder."""
  decoded_img, _ = dcgan.generator(
      codes,
      depth=depth,
      final_size=final_size,
      num_outputs=3,
      is_training=is_training,
      scope='Decoder')

  # Map output to [-1, 1].
  # Use softsign instead of tanh, as per empirical results of
  # http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf.
  return tf.nn.softsign(decoded_img)


def _validate_image_inputs(image_batch):
  image_batch.shape.assert_has_rank(4)
  image_batch.shape[1:].assert_is_fully_defined()


def compression_model(image_batch, num_bits=64, depth=64, is_training=True):
  """Image compression model.

  Args:
    image_batch: A batch of images to compress and reconstruct. Images should
      be normalized already. Shape is [batch, height, width, channels].
    num_bits: Desired number of bits per image in the compressed representation.
    depth: The base number of filters for the encoder and decoder networks.
    is_training: A python bool. If False, run in evaluation mode.

  Returns:
    uncompressed images, binary codes, prebinary codes
  """
  image_batch = tf.convert_to_tensor(image_batch)
  _validate_image_inputs(image_batch)
  final_size = image_batch.shape.as_list()[1]

  prebinary_codes = _encoder(image_batch, is_training, num_bits, depth)
  binary_codes = _binarizer(prebinary_codes, is_training)
  uncompressed_imgs = _decoder(binary_codes, final_size, is_training, depth)
  return uncompressed_imgs, binary_codes, prebinary_codes


def discriminator(image_batch, unused_conditioning=None, depth=64):
  """A thin wrapper around the pix2pix discriminator to conform to TFGAN API."""
  logits, _ = pix2pix.pix2pix_discriminator(
      image_batch, num_filters=[depth, 2 * depth, 4 * depth, 8 * depth])
  return tf.layers.flatten(logits)

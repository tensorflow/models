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
"""Networks for GAN Pix2Pix example using TFGAN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from slim.nets import cyclegan
from slim.nets import pix2pix


def generator(input_images):
  """Thin wrapper around CycleGAN generator to conform to the TFGAN API.

  Args:
    input_images: A batch of images to translate. Images should be normalized
      already. Shape is [batch, height, width, channels].

  Returns:
    Returns generated image batch.
  """
  input_images.shape.assert_has_rank(4)
  with tf.contrib.framework.arg_scope(cyclegan.cyclegan_arg_scope()):
    output_images, _ = cyclegan.cyclegan_generator_resnet(input_images)
  return output_images


def discriminator(image_batch, unused_conditioning=None):
  """A thin wrapper around the Pix2Pix discriminator to conform to TFGAN API."""
  with tf.contrib.framework.arg_scope(pix2pix.pix2pix_arg_scope()):
    logits_4d, _ = pix2pix.pix2pix_discriminator(
        image_batch, num_filters=[64, 128, 256, 512])
    logits_4d.shape.assert_has_rank(4)
  # Output of logits is 4D. Reshape to 2D, for TFGAN.
  logits_2d = tf.contrib.layers.flatten(logits_4d)

  return logits_2d

"""StarGAN data provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data_provider


def provide_data(image_file_patterns, batch_size, patch_size):
  """Data provider wrapper on for the data_provider in gan/cyclegan.

  Args:
    image_file_patterns: A list of file pattern globs.
    batch_size: Python int. Batch size.
    patch_size: Python int. The patch size to extract.

  Returns:
    List of `Tensor` of shape (N, H, W, C) representing the images.
    List of `Tensor` of shape (N, num_domains) representing the labels.
  """

  images = data_provider.provide_custom_data(
      image_file_patterns,
      batch_size=batch_size,
      patch_size=patch_size)

  num_domains = len(images)
  labels = [tf.one_hot([idx] * batch_size, num_domains) for idx in
            range(num_domains)]

  return images, labels

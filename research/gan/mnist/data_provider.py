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
"""Contains code for loading and preprocessing the MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

from slim.datasets import dataset_factory as datasets

slim = tf.contrib.slim


def provide_data(split_name, batch_size, dataset_dir, num_readers=1,
                 num_threads=1):
  """Provides batches of MNIST digits.

  Args:
    split_name: Either 'train' or 'test'.
    batch_size: The number of images in each batch.
    dataset_dir: The directory where the MNIST data can be found.
    num_readers: Number of dataset readers.
    num_threads: Number of prefetching threads.

  Returns:
    images: A `Tensor` of size [batch_size, 28, 28, 1]
    one_hot_labels: A `Tensor` of size [batch_size, mnist.NUM_CLASSES], where
      each row has a single element set to one and the rest set to zeros.
    num_samples: The number of total samples in the dataset.

  Raises:
    ValueError: If `split_name` is not either 'train' or 'test'.
  """
  dataset = datasets.get_dataset('mnist', split_name, dataset_dir=dataset_dir)
  provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=num_readers,
      common_queue_capacity=2 * batch_size,
      common_queue_min=batch_size,
      shuffle=(split_name == 'train'))
  [image, label] = provider.get(['image', 'label'])

  # Preprocess the images.
  image = (tf.to_float(image) - 128.0) / 128.0

  # Creates a QueueRunner for the pre-fetching operation.
  images, labels = tf.train.batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_threads,
      capacity=5 * batch_size)

  one_hot_labels = tf.one_hot(labels, dataset.num_classes)
  return images, one_hot_labels, dataset.num_samples


def float_image_to_uint8(image):
  """Convert float image in [-1, 1) to [0, 255] uint8.

  Note that `1` gets mapped to `0`, but `1 - epsilon` gets mapped to 255.

  Args:
    image: An image tensor. Values should be in [-1, 1).

  Returns:
    Input image cast to uint8 and with integer values in [0, 255].
  """
  image = (image * 128.0) + 128.0
  return tf.cast(image, tf.uint8)

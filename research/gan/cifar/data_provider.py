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
"""Contains code for loading and preprocessing the CIFAR data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from slim.datasets import dataset_factory as datasets

slim = tf.contrib.slim


def provide_data(batch_size, dataset_dir, dataset_name='cifar10',
                 split_name='train', one_hot=True):
  """Provides batches of CIFAR data.

  Args:
    batch_size: The number of images in each batch.
    dataset_dir: The directory where the CIFAR10 data can be found. If `None`,
      use default.
    dataset_name: Name of the dataset.
    split_name: Should be either 'train' or 'test'.
    one_hot: Output one hot vector instead of int32 label.

  Returns:
    images: A `Tensor` of size [batch_size, 32, 32, 3]. Output pixel values are
      in [-1, 1].
    labels: Either (1) one_hot_labels if `one_hot` is `True`
            A `Tensor` of size [batch_size, num_classes], where each row has a
            single element set to one and the rest set to zeros.
            Or (2) labels if `one_hot` is `False`
            A `Tensor` of size [batch_size], holding the labels as integers.
    num_samples: The number of total samples in the dataset.
    num_classes: The number of classes in the dataset.

  Raises:
    ValueError: if the split_name is not either 'train' or 'test'.
  """
  dataset = datasets.get_dataset(
      dataset_name, split_name, dataset_dir=dataset_dir)
  provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      common_queue_capacity=5 * batch_size,
      common_queue_min=batch_size,
      shuffle=(split_name == 'train'))
  [image, label] = provider.get(['image', 'label'])

  # Preprocess the images.
  image = (tf.to_float(image) - 128.0) / 128.0

  # Creates a QueueRunner for the pre-fetching operation.
  images, labels = tf.train.batch(
      [image, label],
      batch_size=batch_size,
      num_threads=32,
      capacity=5 * batch_size)

  labels = tf.reshape(labels, [-1])

  if one_hot:
    labels = tf.one_hot(labels, dataset.num_classes)

  return images, labels, dataset.num_samples, dataset.num_classes


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

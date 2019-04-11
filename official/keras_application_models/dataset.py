# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Prepare dataset for keras model benchmark."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from official.utils.misc import model_helpers  # pylint: disable=g-bad-import-order

# Default values for dataset.
_NUM_CHANNELS = 3
_NUM_CLASSES = 1000


def _get_default_image_size(model):
  """Provide default image size for each model."""
  image_size = (224, 224)
  if model in ["inceptionv3", "xception", "inceptionresnetv2"]:
    image_size = (299, 299)
  elif model in ["nasnetlarge"]:
    image_size = (331, 331)
  return image_size


def generate_synthetic_input_dataset(model, batch_size):
  """Generate synthetic dataset."""
  image_size = _get_default_image_size(model)
  image_shape = (batch_size,) + image_size + (_NUM_CHANNELS,)
  label_shape = (batch_size, _NUM_CLASSES)

  dataset = model_helpers.generate_synthetic_data(
      input_shape=tf.TensorShape(image_shape),
      label_shape=tf.TensorShape(label_shape),
  )
  return dataset


class Cifar10Dataset(object):
  """CIFAR10 dataset, including train and test set.

  Each sample consists of a 32x32 color image, and label is from 10 classes.
  """

  def __init__(self, batch_size):
    """Initializes train/test datasets.

    Args:
      batch_size: int, the number of batch size.
    """
    self.input_shape = (32, 32, 3)
    self.num_classes = 10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)
    y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)
    self.train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(2000).batch(batch_size).repeat()
    self.test_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).shuffle(2000).batch(batch_size).repeat()

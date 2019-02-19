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
"""Prepare cifar10 dataset for keras model testing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class Cifar10Dataset():
  """CIFAR10 dataset, including train and test set.

  Each sample consists of a color image, and label is from 10 classes. The size
  of the image is 32x32 by default.

  The dataset obeys "channel-last" tradition.
  """

  def __init__(self,
               batch_size=32,
               subtract_mean=True,
               data_augmentation=False):
    """Initializes train/test datasets.

    Args:
      batch_size: int, the size of the 1st dimension of the 4D image tensors.
      subtract_mean: boolean, switching on/off mean subtraction normalization.
      data_augmentation: boolean, switching on/off data augmentation for better
          generalization.
    """

    self.image_shape = (32, 32, 3)
    self.num_classes = 10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    self.num_train_examples = len(x_train)
    self.num_test_examples = len(x_test)

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    if subtract_mean:
      x_train_mean = np.mean(x_train, axis=0)
      x_train -= x_train_mean
      x_test -= x_train_mean

    # Convert labels to one-hot vectors
    y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)
    y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)

    # Prepare data preprocessor
    if data_augmentation:
      self._prepare_cifar_data_augmentor()
      self._datagen.fit(x_train)
      data_mapper = lambda x, y: (
          tf.py_function(
              func=self._datagen.flow,
              inp=[x],
              Tout=tf.float32),
          y)
      # TODO(xunkai): Enable data augmentation.
      raise NotImplemented("There's bug in this pipeline.")
    else:
      data_mapper = lambda x, y: (x, y)

    # Build tf.data.Dataset objects
    self.train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(2000)
        .repeat()
        .batch(batch_size)
        .map(data_mapper)
    )
    self.test_dataset = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .shuffle(2000)
        .repeat()
        .batch(batch_size)
    )

  def _prepare_cifar_data_augmentor(self):
    """Prepare CIFAR dataset augmentor and returns the mapping functor."""
    self._datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True)


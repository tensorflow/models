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
"""Prepare cifar10 dataset for keras model performance testing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import tensorflow as tf
import tensorflow_datasets as tfds


class ImageNetDataset():

  def __init__(self):
    builder = tfds.builder("imagenet2012")
    assert builder.info.features["label"].num_classes == 1000
    builder.download_and_prepare()
    assert builder.info.splits["validation"].num_examples == 50000
    self.builder = builder
    self.num_classes = builder.info.features["label"].num_classes
    self.num_train = builder.info.splits["train"].num_examples
    self.num_test = builder.info.splits["validation"].num_examples


  def to_tf_dataset(self, batch_size, image_shape):
    ds = self.builder.as_dataset()
    return (self._preprocess(ds["train"], batch_size, image_shape),
            self._preprocess(ds["validation"], batch_size, image_shape))


  def _preprocess(self, ds, batch_size, image_shape):
    def _convert_example(example):
      label = example["label"]
      label = tf.one_hot(label, self.num_classes, dtype=tf.int64)
      image = example["image"]
      image = tf.dtypes.cast(image, tf.float32)
      image = tf.image.resize(image, image_shape)
      image = image / 127.5 - 1
      return (image, label)

    return ds.map(_convert_example).shuffle(2000).repeat().batch(batch_size)


class Cifar10Dataset():
  """CIFAR10 dataset, including train and test set.

  Each sample consists of a color image, and label is from 10 classes. The size
  of the image is 32x32 by default, however it could be resized by the `dsize`
  parameter.
  """

  @staticmethod
  def resize_images(images, dsize):
    return np.array([cv2.resize(img, dsize) for img in images[:, :, :, :]])

  def __init__(self, dsize=(32, 32), subtract_mean=True):
    """Initializes train/test datasets.

    Args:
      dsize: (int, int), the expected shape of images resized by OpenCV.
    """

    self.input_shape = (dsize[0], dsize[1], 3)
    self.num_classes = 10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    if dsize != (32, 32):
      x_train = Cifar10Dataset.resize_images(x_train, dsize)
      x_test = Cifar10Dataset.resize_images(x_test, dsize)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)
    y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)
    self.num_train_examples = len(x_train)
    self.num_test_examples = len(x_test)
    self.x_train, self.y_train, self.x_test, self.y_test = (
        x_train, y_train, x_test, y_test)
    if subtract_mean:
      x_train_mean = np.mean(self.x_train, axis=0)
      self.x_train -= x_train_mean
      self.x_test -= x_train_mean

  def to_dataset(self, batch_size):
    """Returns tf.data.Dataset version and destroy ndarray version."""

    self.train_dataset = tf.data.Dataset.from_tensor_slices(
        (self.x_train, self.y_train)).shuffle(2000).repeat().batch(batch_size)
    self.test_dataset = tf.data.Dataset.from_tensor_slices(
        (self.x_test, self.y_test)).shuffle(2000).repeat().batch(batch_size)
    del self.x_train, self.y_train, self.x_test, self.y_test
    return (self.train_dataset, self.test_dataset)

  def get_data_augmentor(self, fake=False):
    """Augment cifar10 dataset by cropping and h-flipping."""
    if fake:
      return tf.keras.preprocessing.image.ImageDataGenerator()
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
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
    return datagen

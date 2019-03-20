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

import absl
import tensorflow as tf
import tensorflow_datasets as tfds


class ImageNetDatasetBuilder():
  """A simple wrapper of ImageNet dataset builder."""

  def __init__(self, num_private_threads=None):
    self.builder = tfds.builder("imagenet2012")
    assert self.builder.info.features["label"].num_classes == 1000
    self.builder.download_and_prepare()
    assert self.builder.info.splits["validation"].num_examples == 50000
    self.num_classes = self.builder.info.features["label"].num_classes
    self.num_train = self.builder.info.splits["train"].num_examples
    self.num_test = self.builder.info.splits["validation"].num_examples
    if num_private_threads:
      self.train_ds_options = tf.data.Options()
      self.train_ds_options.experimental_threading.private_threadpool_size = (
          num_private_threads)
    else:
      self.train_ds_options = None


  def to_dataset(self, batch_size, image_shape, take_train_num=-1):
    ds = self.builder.as_dataset()
    train_ds, test_ds = ds["train"], ds["validation"]
    if take_train_num >= 0:
      self.num_train = take_train_num
      absl.logging.info(
          "Train dataset is limited to %d examples." % take_train_num)
    if self.train_ds_options:
      train_ds = train_ds.with_options(self.train_ds_options)
    return (
        self._preprocess_for_train(
            train_ds, batch_size, image_shape, take=take_train_num),
        self._preprocess_for_eval(test_ds, batch_size, image_shape)
    )


  def _preprocess_for_train(self, ds, batch_size, image_shape, take=-1):
    # Take the whole image to distort
    default_bbox = tf.constant([0, 0, 1, 1], dtype=tf.float32, shape=[1, 1, 4])
    def _convert_example(example):
      label = example["label"]
      label = tf.one_hot(label, self.num_classes, dtype=tf.int64)
      image = example["image"]
      # Normalization
      image = tf.dtypes.cast(image, tf.float32)

      image = image / 255.0  # Image space: [0, 1]
      # Image distortion
      begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
          image_size=tf.shape(image),
          bounding_boxes=default_bbox,
          min_object_covered=0.1,
          aspect_ratio_range=(0.75, 1.33),
          area_range=(0.05, 1.0),
          max_attempts=100)
      distorted_image = tf.slice(image, begin, size)
      resized_image = tf.image.resize(distorted_image, image_shape)
      flipped_image= tf.image.random_flip_left_right(resized_image)
      image = flipped_image * 2 - 1  # Image space: [-1, 1]
      return (image, label)

    buffer_size = 10000
    ds = ds.take(take).shuffle(buffer_size).repeat().prefetch(buffer_size)
    return ds.map(_convert_example).batch(batch_size)


  def _preprocess_for_eval(self, ds, batch_size, image_shape):
    def _convert_example(example):
      label = example["label"]
      label = tf.one_hot(label, self.num_classes, dtype=tf.int64)
      image = example["image"]
      image = tf.image.resize(image, image_shape)
      image = image / 127.5 - 1  # Image space: [-1, 1]
      return (image, label)
    return ds.map(_convert_example).repeat().batch(batch_size)

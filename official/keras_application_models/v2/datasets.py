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

  def __init__(self):
    self.builder = tfds.builder("imagenet2012")
    assert self.builder.info.features["label"].num_classes == 1000
    self.builder.download_and_prepare()
    assert self.builder.info.splits["validation"].num_examples == 50000
    self.num_classes = self.builder.info.features["label"].num_classes
    self.num_train = self.builder.info.splits["train"].num_examples
    self.num_test = self.builder.info.splits["validation"].num_examples


  def to_dataset(self, batch_size, image_shape, take_train_num=-1):
    ds = self.builder.as_dataset()
    if take_train_num >= 0:
      self.num_train = take_train_num
      absl.logging.info(
          "Train dataset is limited to %d examples." % take_train_num)
    return (
        self._preprocess(
            ds["train"], batch_size, image_shape, take=take_train_num),
        self._preprocess(ds["validation"], batch_size, image_shape)
    )


  def _preprocess(self, ds, batch_size, image_shape, take=-1):
    def _convert_example(example):
      label = example["label"]
      label = tf.one_hot(label, self.num_classes, dtype=tf.int64)
      image = example["image"]
      image = tf.dtypes.cast(image, tf.float32)
      image = tf.image.resize(image, image_shape)
      image = image / 127.5 - 1
      return (image, label)

    return ds.take(take).map(_convert_example).repeat().batch(batch_size)


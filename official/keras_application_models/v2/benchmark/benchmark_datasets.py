# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Dataset interface for PerfZero benchmarks."""
import tensorflow as tf


from official.resnet import imagenet_main
from official.resnet import cifar10_main


class DatasetBuilderBase():

  def __init__(self, input_fn, num_dataset_private_threads=None):
    self._num_dataset_private_threads = num_dataset_private_threads
    self._input_fn = input_fn

  def to_dataset(self, batch_size, image_shape, take_train_num=-1):
    # N.B: -1 // x == -1, if x is a positive integer
    if take_train_num >= 0:
      self.num_train = take_train_num // batch_size * batch_size
    return (
        self._get_perfzero_dataset(True, batch_size).take(
            take_train_num // batch_size).repeat(),
        self._get_perfzero_dataset(False, batch_size))

  def _get_perfzero_dataset(self, is_training, batch_size):
    def _to_categorical(image, label):
      label = tf.one_hot(label, self.num_classes, dtype=tf.int64)
      return (image, label)

    raw_dataset = self._input_fn(
        is_training=is_training,
        data_dir=self._data_dir,
        batch_size=batch_size,
        num_epochs=None,  # repeat infinitely
        datasets_num_private_threads=self._num_dataset_private_threads)

    return raw_dataset.map(
        _to_categorical, num_parallel_calls=tf.data.experimental.AUTOTUNE)


class ImageNetDatasetBuilder(DatasetBuilderBase):
  """Wrapper for accessing ImageNet with PerfZero."""

  def __init__(self, data_dir, num_dataset_private_threads=None):
    self.num_classes = 1000
    self.num_train = 1281167
    self.num_test = 50000
    self._data_dir = data_dir
    super().__init__(imagenet_main.input_fn, num_dataset_private_threads)


class Cifar10DatasetBuilder(DatasetBuilderBase):
  """Wrapper for accessing Cifar10 with PerfZero."""

  def __init__(self, data_dir, num_dataset_private_threads=None):
    self.num_classes = 10
    self.num_train = 50000
    self.num_test = 10000
    self._data_dir = data_dir
    super().__init__(cifar10_main.input_fn, num_dataset_private_threads)


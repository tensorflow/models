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
from official.resnet.keras import keras_imagenet_main


class ImageNetDatasetBuilder():
  """Wrapper for accessing ImageNet with PerfZero."""

  def __init__(self, data_dir, num_dataset_private_threads=None):
    self.num_classes = 1000
    self.num_train = 1281167
    self.num_test = 50000
    self._data_dir = data_dir
    self._num_dataset_private_threads = num_dataset_private_threads

  def to_dataset(self, batch_size, image_shape, take_train_num=-1):
    return (
        self._get_perfzero_dataset(True, batch_size).take(take_train_num),
        self._get_perfzero_dataset(False, batch_size))

  def _get_perfzero_dataset(self, is_training, batch_size):
    return imagenet_main.input_fn(
        is_training=is_training,
        data_dir=self._data_dir,
        batch_size=batch_size,
        num_epochs=None,  # repeat infinitely
        parse_record_fn=keras_imagenet_main.parse_record_keras,
        datasets_num_private_threads=self._num_dataset_private_threads)


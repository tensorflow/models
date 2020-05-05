# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Runs an Image Classification task for MobileNet."""

from typing import Mapping, Text, Any

import logging
import tensorflow as tf

from research.mobilenet.dataset_loader import load_tfds, pipeline
from research.mobilenet.dataset_loader import ImageNetConfig


def _get_metrics(one_hot: bool) -> Mapping[Text, Any]:
  """Get a dict of available metrics to track."""
  if one_hot:
    return {
      # (name, metric_fn)
      'acc': tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
      'accuracy': tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
      'top_1': tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
      'top_5': tf.keras.metrics.TopKCategoricalAccuracy(
        k=5, name='top_5_accuracy'),
    }
  else:
    return {
      # (name, metric_fn)
      'acc': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
      'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
      'top_1': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
      'top_5': tf.keras.metrics.SparseTopKCategoricalAccuracy(
        k=5, name='top_5_accuracy'),
    }


def get_args():
  """Parse command line arguments"""
  pass


def get_dataset():
  """Build dataset for training, evaluation and test"""
  pass


def build_model():
  """Build mobilenet model given configuration"""


def train_and_eval():
  """Runs the train and eval path using compile/fit."""
  pass


def main():
  pass


if __name__ == '__main__':
  logging.basicConfig(
    format='%(asctime)-15s:%(levelname)s:%(module)s:%(message)s',
    level=logging.INFO)

  main()

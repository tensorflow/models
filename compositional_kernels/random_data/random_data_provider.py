# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Contains code for generating random data.
"""

import numpy as np
import tensorflow as tf
from base import config

X_DIM = 28
Y_DIM = 1
NUM_CHANNELS = 1
NUM_CLASSES = 10
NUM_TRAINING_EXAMPLES = 512
NUM_TEST_EXAMPLES = 128


def GetRandomConfig():
  random_config = config.LearningParams()
  random_config.SetValue('number_of_classes', NUM_CLASSES)
  random_config.SetValue('number_of_examples', NUM_TRAINING_EXAMPLES)
  random_config.SetValue('number_of_test_examples', NUM_TEST_EXAMPLES)
  return random_config


class Random_Input(object):

  def __init__(self):
    self.generate_data()

  def generate_data(self):
    self.train_examples = np.random.randn(NUM_TRAINING_EXAMPLES,
                                          X_DIM,
                                          Y_DIM,
                                          NUM_CHANNELS).astype(np.float32)
    self.train_labels = np.random.randint(NUM_CLASSES,
                                          size=NUM_TRAINING_EXAMPLES)

    self.test_examples = np.random.randn(NUM_TEST_EXAMPLES,
                                         X_DIM,
                                         Y_DIM,
                                         NUM_CHANNELS).astype(np.float32)
    self.test_labels = np.random.randint(NUM_CLASSES,
                                         size=NUM_TEST_EXAMPLES)

  def Shuffle(self, indices, current_index):
    indices_update = tf.assign(indices, tf.random_shuffle(indices))
    current_index_update = tf.assign(current_index, 0)
    return current_index_update, indices_update

  def ProvideData(self, batch_size, split_name):
    if split_name == 'train':
      all_examples = tf.constant(self.train_examples)
      all_labels = tf.constant(self.train_labels)
      num_examples = NUM_TRAINING_EXAMPLES
      all_size = tf.constant(num_examples)
    elif split_name == 'test':
      all_examples = tf.constant(self.test_examples)
      all_labels = tf.constant(self.test_labels)
      num_examples = NUM_TEST_EXAMPLES
      all_size = tf.constant(num_examples)
    else:
      raise ValueError('split %s not recognized', split_name)

    current_index = tf.Variable(num_examples, dtype=tf.int32)
    indices = tf.Variable(tf.constant(np.arange(num_examples)),
                          dtype=tf.int64)

    start, ind = tf.cond(current_index + batch_size < all_size,
                         lambda: (current_index, indices),
                         lambda: self.Shuffle(indices, current_index))

    batch = tf.slice(ind, [start], [batch_size])

    with tf.control_dependencies([batch]):
      current_index_next = current_index.assign_add(batch_size)
      with tf.control_dependencies([current_index_next]):
        example_batch = tf.gather(all_examples, batch)
        labels = tf.gather(all_labels, batch)

    return example_batch, tf.one_hot(labels, NUM_CLASSES)

# Copyright 2020 The Orbit Authors. All Rights Reserved.
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
"""Tests for orbit.standard_runner."""

from orbit import standard_runner

import tensorflow as tf


def dataset_fn(input_context=None):
  del input_context

  def dummy_data(_):
    return tf.zeros((1, 1), dtype=tf.float32)

  dataset = tf.data.Dataset.range(1)
  dataset = dataset.repeat()
  dataset = dataset.map(
      dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


class TestRunner(standard_runner.StandardTrainer,
                 standard_runner.StandardEvaluator):
  """Implements the training and evaluation APIs for tests."""

  def __init__(self):
    self.strategy = tf.distribute.get_strategy()
    self.global_step = tf.Variable(
        0,
        trainable=False,
        dtype=tf.int64,
        name='global_step',
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
    standard_runner.StandardTrainer.__init__(self, train_dataset=None)
    standard_runner.StandardEvaluator.__init__(self, eval_dataset=None)

  def train_loop_begin(self):
    self.train_dataset = (
        self.strategy.experimental_distribute_datasets_from_function(dataset_fn)
    )

  def train_step(self, iterator):

    def _replicated_step(_):
      self.global_step.assign_add(1)

    self.strategy.run(_replicated_step, args=(next(iterator),))

  def train_loop_end(self):
    return self.global_step.numpy()

  def eval_begin(self):
    self.eval_dataset = self.strategy.experimental_distribute_datasets_from_function(
        dataset_fn)

  def eval_step(self, iterator):

    def _replicated_step(_):
      self.global_step.assign_add(1)

    self.strategy.run(_replicated_step, args=(next(iterator),))

  def eval_end(self):
    return self.global_step.numpy()


class StandardRunnerTest(tf.test.TestCase):

  def test_train(self):
    test_runner = TestRunner()
    self.assertEqual(
        test_runner.train(tf.convert_to_tensor(10, dtype=tf.int32)), 10)

  def test_eval(self):
    test_runner = TestRunner()
    self.assertEqual(
        test_runner.evaluate(tf.convert_to_tensor(10, dtype=tf.int32)), 10)


if __name__ == '__main__':
  tf.test.main()

# Copyright 2024 The Orbit Authors. All Rights Reserved.
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

"""Tests for orbit.standard_runner."""

from absl.testing import parameterized

from orbit import standard_runner
from orbit import utils

import tensorflow as tf, tf_keras


def dataset_fn(input_context=None):
  del input_context

  def dummy_data(_):
    return tf.zeros((1, 1), dtype=tf.float32)

  dataset = tf.data.Dataset.range(1)
  dataset = dataset.repeat()
  dataset = dataset.map(
      dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


class TestTrainer(standard_runner.StandardTrainer):
  """A StandardTrainer subclass for tests."""

  def __init__(self, options=None):
    self.strategy = tf.distribute.get_strategy()
    self.global_step = utils.create_global_step()
    dataset = self.strategy.distribute_datasets_from_function(dataset_fn)
    super().__init__(train_dataset=dataset, options=options)

  def train_loop_begin(self):
    self.global_step.assign(0)

  def train_step(self, iterator):

    def replica_step(_):
      self.global_step.assign_add(1)

    self.strategy.run(replica_step, args=(next(iterator),))

  def train_loop_end(self):
    return self.global_step.numpy()


class TestEvaluator(standard_runner.StandardEvaluator):
  """A StandardEvaluator subclass for tests."""

  def __init__(self, options=None):
    self.strategy = tf.distribute.get_strategy()
    self.global_step = utils.create_global_step()
    dataset = self.strategy.distribute_datasets_from_function(dataset_fn)
    super().__init__(eval_dataset=dataset, options=options)

  def eval_begin(self):
    self.global_step.assign(0)

  def eval_step(self, iterator):

    def replica_step(_):
      self.global_step.assign_add(1)

    self.strategy.run(replica_step, args=(next(iterator),))

  def eval_end(self):
    return self.global_step.numpy()


class TestEvaluatorWithOutputsAggregation(standard_runner.StandardEvaluator):
  """A StandardEvaluator subclass for tests."""

  def __init__(self, options=None):
    self.strategy = tf.distribute.get_strategy()
    dataset = self.strategy.distribute_datasets_from_function(
        lambda _: tf.data.Dataset.range(10))
    super().__init__(eval_dataset=dataset, options=options)

  def eval_begin(self):
    return {"logits": tf.constant((0.0,))}

  def eval_reduce(self, state, step_outputs):
    state["logits"] = tf.concat([state["logits"], step_outputs], 0)
    return state

  def eval_step(self, iterator):

    def replica_step(x):
      x = tf.cast(x, tf.float32)
      return tf.reduce_sum(x)

    return self.strategy.experimental_local_results(
        self.strategy.run(replica_step, args=(next(iterator),)))

  def eval_end(self, outputs):
    return tf.reduce_sum(outputs["logits"])


class StandardRunnerTest(parameterized.TestCase):

  def test_default_trainer(self):
    trainer = TestTrainer()
    self.assertEqual(trainer.train(tf.constant(10)), 10)

  def test_trainer_with_tpu_summary_optimization(self):
    options = standard_runner.StandardTrainerOptions(
        use_tpu_summary_optimization=True)
    trainer = TestTrainer(options)
    self.assertEqual(trainer.train(tf.constant(10)), 10)

  @parameterized.named_parameters(("use_tf_while_loop", True), ("", False))
  def test_default_evaluator(self, use_tf_while_loop):
    options = standard_runner.StandardEvaluatorOptions(
        use_tf_while_loop=use_tf_while_loop)
    evaluator = TestEvaluator(options)
    self.assertEqual(evaluator.evaluate(tf.constant(10)), 10)

  @parameterized.named_parameters(("use_tf_while_loop", True), ("", False))
  def test_evaluator_with_outputs_aggregation(self, use_tf_while_loop):
    options = standard_runner.StandardEvaluatorOptions(
        use_tf_while_loop=use_tf_while_loop)
    evaluator = TestEvaluatorWithOutputsAggregation(options)
    self.assertEqual(evaluator.evaluate(tf.constant(10)), 45)

  @parameterized.named_parameters(
      ("recreate_iterator_for_each_eval", True, 10, 10),
      ("not_recreate_iterator_for_each_eval", False, 10, 35))
  def test_evaluator_with_repeat_dataset(self, recreate_iterator_for_each_eval,
                                         sum_for_1st_time, sum_for_2nd_time):
    options = standard_runner.StandardEvaluatorOptions(
        recreate_iterator_for_each_eval=recreate_iterator_for_each_eval)
    evaluator = TestEvaluatorWithOutputsAggregation(options)
    self.assertEqual(evaluator.evaluate(tf.constant(5)), sum_for_1st_time)
    self.assertEqual(evaluator.evaluate(tf.constant(5)), sum_for_2nd_time)


if __name__ == "__main__":
  tf.test.main()

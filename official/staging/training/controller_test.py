# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for official.staging.training.controller."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.staging.training import controller
from official.staging.training import standard_runnable


def all_strategy_combinations():
  """Gets combinations of distribution strategies."""
  return combinations.combine(
      strategy=[
          strategy_combinations.one_device_strategy,
          strategy_combinations.tpu_strategy,
          strategy_combinations.one_device_strategy_gpu,
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
      ],
      mode="eager",
  )


def create_model():
  x = tf.keras.layers.Input(shape=(3,), name="input")
  y = tf.keras.layers.Dense(4, name="dense")(x)
  model = tf.keras.Model(x, y)
  return model


def summaries_with_matching_keyword(keyword, summary_dir):
  """Yields summary protos matching given keyword from event file."""
  event_paths = tf.io.gfile.glob(os.path.join(summary_dir, "events*"))
  for event in tf.compat.v1.train.summary_iterator(event_paths[-1]):
    if event.summary is not None:
      for value in event.summary.value:
        if keyword in value.tag:
          tf.compat.v1.logging.error(event)
          yield event.summary


def check_eventfile_for_keyword(keyword, summary_dir):
  """Checks event files for the keyword."""
  return any(summaries_with_matching_keyword(keyword, summary_dir))


def dataset_fn(ctx):
  del ctx
  inputs = np.zeros((10, 3), dtype=np.float32)
  targets = np.zeros((10, 4), dtype=np.float32)
  dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
  dataset = dataset.repeat(100)
  dataset = dataset.batch(10, drop_remainder=True)
  return dataset


class TestRunnable(standard_runnable.StandardTrainable,
                   standard_runnable.StandardEvaluable):
  """Implements the training and evaluation APIs for the test model."""

  def __init__(self):
    standard_runnable.StandardTrainable.__init__(self)
    standard_runnable.StandardEvaluable.__init__(self)
    self.strategy = tf.distribute.get_strategy()
    self.model = create_model()
    self.optimizer = tf.keras.optimizers.RMSprop()
    self.global_step = self.optimizer.iterations
    self.train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
    self.eval_loss = tf.keras.metrics.Mean("eval_loss", dtype=tf.float32)

  def build_train_dataset(self):
    return self.strategy.experimental_distribute_datasets_from_function(
        dataset_fn)

  def train_step(self, iterator):

    def _replicated_step(inputs):
      """Replicated training step."""
      inputs, targets = inputs
      with tf.GradientTape() as tape:
        outputs = self.model(inputs)
        loss = tf.math.reduce_sum(outputs - targets)
      grads = tape.gradient(loss, self.model.variables)
      self.optimizer.apply_gradients(zip(grads, self.model.variables))
      self.train_loss.update_state(loss)

    self.strategy.run(_replicated_step, args=(next(iterator),))

  def train_loop_end(self):
    return {
        "loss": self.train_loss.result(),
    }

  def build_eval_dataset(self):
    return self.strategy.experimental_distribute_datasets_from_function(
        dataset_fn)

  def eval_begin(self):
    self.eval_loss.reset_states()

  def eval_step(self, iterator):

    def _replicated_step(inputs):
      """Replicated evaluation step."""
      inputs, targets = inputs
      outputs = self.model(inputs)
      loss = tf.math.reduce_sum(outputs - targets)
      self.eval_loss.update_state(loss)

    self.strategy.run(_replicated_step, args=(next(iterator),))

  def eval_end(self):
    return {
        "eval_loss": self.eval_loss.result(),
    }


class ControllerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ControllerTest, self).setUp()
    self.model_dir = self.get_temp_dir()

  def test_no_checkpoint(self):
    test_runnable = TestRunnable()
    # No checkpoint manager and no strategy.
    test_controller = controller.Controller(
        train_fn=test_runnable.train,
        eval_fn=test_runnable.evaluate,
        global_step=test_runnable.global_step,
        train_steps=10,
        steps_per_loop=2,
        summary_dir=os.path.join(self.model_dir, "summaries/train"),
        summary_interval=2,
        eval_summary_dir=os.path.join(self.model_dir, "summaries/eval"),
        eval_steps=2,
        eval_interval=5)
    test_controller.train(evaluate=True)
    self.assertEqual(test_runnable.global_step.numpy(), 10)
    # Loss and accuracy values should be written into summaries.
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(self.model_dir, "summaries/train")))
    self.assertTrue(
        check_eventfile_for_keyword(
            "loss", os.path.join(self.model_dir, "summaries/train")))
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(self.model_dir, "summaries/eval")))
    self.assertTrue(
        check_eventfile_for_keyword(
            "eval_loss", os.path.join(self.model_dir, "summaries/eval")))
    # No checkpoint, so global step starts from 0.
    test_runnable.global_step.assign(0)
    test_controller.train(evaluate=True)
    self.assertEqual(test_runnable.global_step.numpy(), 10)

  def test_no_checkpoint_and_summaries(self):
    test_runnable = TestRunnable()
    # No checkpoint + summary directories.
    test_controller = controller.Controller(
        train_fn=test_runnable.train,
        eval_fn=test_runnable.evaluate,
        global_step=test_runnable.global_step,
        train_steps=10,
        steps_per_loop=2,
        eval_steps=2,
        eval_interval=5)
    test_controller.train(evaluate=True)
    self.assertEqual(test_runnable.global_step.numpy(), 10)

  @combinations.generate(all_strategy_combinations())
  def test_train_and_evaluate(self, strategy):
    with strategy.scope():
      test_runnable = TestRunnable()

    checkpoint = tf.train.Checkpoint(
        model=test_runnable.model, optimizer=test_runnable.optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        self.model_dir,
        max_to_keep=None,
        step_counter=test_runnable.global_step,
        checkpoint_interval=10)
    test_controller = controller.Controller(
        strategy=strategy,
        train_fn=test_runnable.train,
        eval_fn=test_runnable.evaluate,
        global_step=test_runnable.global_step,
        train_steps=10,
        steps_per_loop=2,
        summary_dir=os.path.join(self.model_dir, "summaries/train"),
        summary_interval=2,
        checkpoint_manager=checkpoint_manager,
        eval_summary_dir=os.path.join(self.model_dir, "summaries/eval"),
        eval_steps=2,
        eval_interval=5)
    test_controller.train(evaluate=True)

    # Checkpoints are saved.
    self.assertNotEmpty(tf.io.gfile.glob(os.path.join(self.model_dir, "ckpt*")))

    # Loss and accuracy values should be written into summaries.
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(self.model_dir, "summaries/train")))
    self.assertTrue(
        check_eventfile_for_keyword(
            "loss", os.path.join(self.model_dir, "summaries/train")))
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(self.model_dir, "summaries/eval")))
    self.assertTrue(
        check_eventfile_for_keyword(
            "eval_loss", os.path.join(self.model_dir, "summaries/eval")))

  @combinations.generate(all_strategy_combinations())
  def test_train_only(self, strategy):
    with strategy.scope():
      test_runnable = TestRunnable()

    checkpoint = tf.train.Checkpoint(
        model=test_runnable.model, optimizer=test_runnable.optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        self.model_dir,
        max_to_keep=None,
        step_counter=test_runnable.global_step,
        checkpoint_interval=10)
    test_controller = controller.Controller(
        strategy=strategy,
        train_fn=test_runnable.train,
        global_step=test_runnable.global_step,
        train_steps=10,
        steps_per_loop=2,
        summary_dir=os.path.join(self.model_dir, "summaries/train"),
        summary_interval=2,
        checkpoint_manager=checkpoint_manager,
        eval_summary_dir=os.path.join(self.model_dir, "summaries/eval"),
    )
    test_controller.train(evaluate=False)

    # Checkpoints are saved.
    self.assertNotEmpty(tf.io.gfile.glob(os.path.join(self.model_dir, "ckpt*")))

    # Only train summaries are written.
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(self.model_dir, "summaries/train")))
    self.assertTrue(
        check_eventfile_for_keyword(
            "loss", os.path.join(self.model_dir, "summaries/train")))
    self.assertFalse(
        tf.io.gfile.exists(os.path.join(self.model_dir, "summaries/eval")))

  @combinations.generate(all_strategy_combinations())
  def test_evaluate_only(self, strategy):
    with strategy.scope():
      test_runnable = TestRunnable()

    checkpoint = tf.train.Checkpoint(model=test_runnable.model)
    checkpoint.save(os.path.join(self.model_dir, "ckpt"))

    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        self.model_dir,
        max_to_keep=None,
        step_counter=test_runnable.global_step)
    test_controller = controller.Controller(
        strategy=strategy,
        eval_fn=test_runnable.evaluate,
        global_step=test_runnable.global_step,
        checkpoint_manager=checkpoint_manager,
        summary_dir=os.path.join(self.model_dir, "summaries/train"),
        eval_summary_dir=os.path.join(self.model_dir, "summaries/eval"),
        eval_steps=2,
        eval_interval=5)
    test_controller.evaluate()

    # Only eval summaries are written
    self.assertFalse(
        tf.io.gfile.exists(os.path.join(self.model_dir, "summaries/train")))
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(self.model_dir, "summaries/eval")))
    self.assertTrue(
        check_eventfile_for_keyword(
            "eval_loss", os.path.join(self.model_dir, "summaries/eval")))


if __name__ == "__main__":
  tf.test.main()

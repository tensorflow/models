# Copyright 2025 The Orbit Authors. All Rights Reserved.
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

"""Tests for orbit.controller."""

import os

from absl import logging
from absl.testing import parameterized

import numpy as np

from orbit import controller
from orbit import runner
from orbit import standard_runner
import orbit.utils

import tensorflow as tf, tf_keras


def create_model():
  x = tf_keras.layers.Input(shape=(3,), name="input")
  y = tf_keras.layers.Dense(4, name="dense")(x)
  model = tf_keras.Model(x, y)
  return model


def summaries_with_matching_keyword(keyword, summary_dir):
  """Returns summary protos matching given keyword from event file."""
  matches = []
  event_paths = tf.io.gfile.glob(os.path.join(summary_dir, "events*"))
  for event in tf.compat.v1.train.summary_iterator(event_paths[-1]):
    if event.summary is not None:
      for value in event.summary.value:
        if keyword in value.tag:
          matches.append(event.summary)
  return matches


def dataset_fn(ctx):
  del ctx
  inputs = np.zeros((10, 3), dtype=np.float32)
  targets = np.ones((10, 4), dtype=np.float32)
  dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
  dataset = dataset.repeat(100)
  dataset = dataset.batch(10, drop_remainder=True)
  return dataset


class TestRunner(standard_runner.StandardTrainer,
                 standard_runner.StandardEvaluator):
  """Implements the training and evaluation APIs for the test model."""

  def __init__(self, return_numpy=False):
    self.strategy = tf.distribute.get_strategy()
    self.model = create_model()
    self.optimizer = tf_keras.optimizers.RMSprop(learning_rate=0.1)
    self.global_step = self.optimizer.iterations
    self.train_loss = tf_keras.metrics.Mean("train_loss", dtype=tf.float32)
    self.eval_loss = tf_keras.metrics.Mean("eval_loss", dtype=tf.float32)
    self.return_numpy = return_numpy
    train_dataset = self.strategy.distribute_datasets_from_function(dataset_fn)
    eval_dataset = self.strategy.distribute_datasets_from_function(dataset_fn)
    standard_runner.StandardTrainer.__init__(self, train_dataset)
    standard_runner.StandardEvaluator.__init__(self, eval_dataset)

  def train_step(self, iterator):

    def _replicated_step(inputs):
      """Replicated training step."""
      inputs, targets = inputs
      with tf.GradientTape() as tape:
        outputs = self.model(inputs)
        loss = tf.reduce_mean(tf_keras.losses.MSE(targets, outputs))
      grads = tape.gradient(loss, self.model.variables)
      self.optimizer.apply_gradients(zip(grads, self.model.variables))
      self.train_loss.update_state(loss)

    self.strategy.run(_replicated_step, args=(next(iterator),))

  def train_loop_end(self):
    train_loss = self.train_loss.result()
    return {
        "loss": train_loss.numpy() if self.return_numpy else train_loss,
    }

  def build_eval_dataset(self):
    return self.strategy.distribute_datasets_from_function(dataset_fn)

  def eval_begin(self):
    self.eval_loss.reset_states()

  def eval_step(self, iterator):

    def _replicated_step(inputs):
      """Replicated evaluation step."""
      inputs, targets = inputs
      outputs = self.model(inputs)
      loss = tf.reduce_mean(tf_keras.losses.MSE(targets, outputs))
      self.eval_loss.update_state(loss)

    self.strategy.run(_replicated_step, args=(next(iterator),))

  def eval_end(self):
    eval_loss = self.eval_loss.result()
    return {
        "eval_loss": eval_loss.numpy() if self.return_numpy else eval_loss,
    }


class TestEvaluator(standard_runner.StandardEvaluator):
  """Implements the training and evaluation APIs for the test model."""

  def __init__(self):
    self.strategy = tf.distribute.get_strategy()
    self.model = create_model()
    eval_dataset = self.strategy.distribute_datasets_from_function(dataset_fn)
    standard_runner.StandardEvaluator.__init__(self, eval_dataset)

  def eval_reduce(self, state, output):
    state.append(output)
    return state

  def eval_begin(self):
    return []

  def eval_step(self, iterator):

    def _replicated_step(inputs):
      """Replicated evaluation step."""
      inputs, targets = inputs
      outputs = self.model(inputs)
      loss = tf.reduce_mean(tf_keras.losses.MSE(targets, outputs))
      return loss

    per_replica_losses = self.strategy.run(
        _replicated_step, args=(next(iterator),))
    mean_loss = self.strategy.reduce(
        tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    return mean_loss

  def eval_end(self, outputs):
    return {
        "eval_loss": tf.reduce_mean(outputs),
    }


class TestEvaluatorNoOutput(runner.AbstractEvaluator):

  def evaluate(self, num_steps):
    pass


class TestEvaluatorWithNestedSummary(standard_runner.StandardEvaluator):
  """Implements the training and evaluation APIs for the test model."""

  def __init__(self):
    self.strategy = tf.distribute.get_strategy()
    self.model = create_model()
    dataset = self.strategy.distribute_datasets_from_function(dataset_fn)
    dataset2 = self.strategy.distribute_datasets_from_function(dataset_fn)
    self.loss = tf_keras.metrics.Mean("loss", dtype=tf.float32)
    self.accuracy = tf_keras.metrics.CategoricalAccuracy(
        "accuracy", dtype=tf.float32)
    self.loss2 = tf_keras.metrics.Mean("loss", dtype=tf.float32)
    self.accuracy2 = tf_keras.metrics.CategoricalAccuracy(
        "accuracy", dtype=tf.float32)
    standard_runner.StandardEvaluator.__init__(
        self, eval_dataset={
            "dataset": dataset,
            "dataset2": dataset2
        })

  def eval_step(self, iterator):

    def _replicated_step(loss, accuracy, inputs):
      """Replicated evaluation step."""
      inputs, targets = inputs
      outputs = self.model(inputs)
      loss.update_state(tf_keras.losses.MSE(targets, outputs))
      accuracy.update_state(targets, outputs)

    self.strategy.run(
        lambda inputs: _replicated_step(self.loss, self.accuracy, inputs),
        args=(next(iterator["dataset"]),))
    self.strategy.run(
        lambda inputs: _replicated_step(self.loss2, self.accuracy2, inputs),
        args=(next(iterator["dataset2"]),))

  def eval_end(self):
    return {
        "dataset": {
            "loss": self.loss.result(),
            "accuracy": self.accuracy.result()
        },
        "dataset2": {
            "loss": self.loss2.result(),
            "accuracy": self.accuracy2.result()
        },
    }


class TestTrainerWithSummaries(standard_runner.StandardTrainer):
  """A Trainer model with summaries for testing purposes."""

  def __init__(self):
    self.strategy = tf.distribute.get_strategy()
    self.model = create_model()
    self.optimizer = tf_keras.optimizers.RMSprop(learning_rate=0.1)
    self.global_step = self.optimizer.iterations
    self.train_loss = tf_keras.metrics.Mean("train_loss", dtype=tf.float32)
    train_dataset = self.strategy.distribute_datasets_from_function(dataset_fn)
    standard_runner.StandardTrainer.__init__(
        self,
        train_dataset,
        options=standard_runner.StandardTrainerOptions(
            use_tpu_summary_optimization=True))

  def build_train_dataset(self):
    return self.strategy.distribute_datasets_from_function(dataset_fn)

  def train_step(self, iterator):

    def _replicated_step(inputs):
      """Replicated training step."""
      inputs, targets = inputs
      with tf.GradientTape() as tape:
        outputs = self.model(inputs)
        loss = tf.reduce_mean(tf_keras.losses.MSE(targets, outputs))
      tf.summary.scalar("loss", loss)
      grads = tape.gradient(loss, self.model.variables)
      self.optimizer.apply_gradients(zip(grads, self.model.variables))
      self.train_loss.update_state(loss)

    self.strategy.run(_replicated_step, args=(next(iterator),))


class ControllerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.model_dir = self.get_temp_dir()

  def test_no_checkpoint(self):
    test_runner = TestRunner()
    # No checkpoint manager and no strategy.
    test_controller = controller.Controller(
        trainer=test_runner,
        evaluator=test_runner,
        global_step=test_runner.global_step,
        steps_per_loop=2,
        summary_dir=os.path.join(self.model_dir, "summaries/train"),
        eval_summary_dir=os.path.join(self.model_dir, "summaries/eval"))
    test_controller.train_and_evaluate(
        train_steps=10, eval_steps=2, eval_interval=6)
    self.assertEqual(test_runner.global_step, 10)
    # Loss and accuracy values should be written into summaries.
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(self.model_dir, "summaries/train")))
    self.assertNotEmpty(
        summaries_with_matching_keyword(
            "loss", os.path.join(self.model_dir, "summaries/train")))
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(self.model_dir, "summaries/eval")))
    self.assertNotEmpty(
        summaries_with_matching_keyword(
            "eval_loss", os.path.join(self.model_dir, "summaries/eval")))
    # No checkpoint, so global step starts from 0.
    test_runner.global_step.assign(0)
    test_controller.train_and_evaluate(
        train_steps=10, eval_steps=2, eval_interval=6)
    self.assertEqual(test_runner.global_step, 10)
    self.assertTrue(controller._orbit_api_gauge.get_cell().value())

  def test_no_checkpoint_and_summaries(self):
    test_runner = TestRunner()
    # No checkpoint + summary directories.
    test_controller = controller.Controller(
        trainer=test_runner,
        evaluator=test_runner,
        global_step=test_runner.global_step,
        steps_per_loop=2)
    test_controller.train_and_evaluate(
        train_steps=10, eval_steps=2, eval_interval=6)
    self.assertEqual(test_runner.global_step, 10)
    self.assertTrue(controller._orbit_api_gauge.get_cell().value())

  @parameterized.named_parameters(
      ("_sync_checkpoint_saving", False),
      ("_async_checkpoint_saving", True)
  )
  def test_has_checkpoint_no_summaries(self, enable_async_checkpoint_saving):
    test_runner = TestRunner()
    # Has checkpoint, but no summary directories.
    checkpoint = tf.train.Checkpoint(model=test_runner.model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        self.model_dir,
        max_to_keep=None,
        step_counter=test_runner.global_step)
    test_controller = controller.Controller(
        trainer=test_runner,
        evaluator=test_runner,
        global_step=test_runner.global_step,
        checkpoint_manager=checkpoint_manager,
        enable_async_checkpointing=enable_async_checkpoint_saving,
        steps_per_loop=2)
    test_controller.train_and_evaluate(
        train_steps=10, eval_steps=2, eval_interval=6)
    self.assertEqual(test_runner.global_step, 10)
    self.assertTrue(controller._orbit_api_gauge.get_cell().value())

    # No summaries are saved.
    self.assertEmpty(tf.io.gfile.glob(
        os.path.join(checkpoint_manager.directory, "events.*")))

  @parameterized.named_parameters(
      ("_sync_checkpoint_saving", False),
      ("_async_checkpoint_saving", True)
  )
  def test_has_checkpoint_eval_summary_only(
      self, enable_async_checkpoint_saving
  ):
    test_runner = TestRunner()
    # Has checkpoint, but no summary directories.
    checkpoint = tf.train.Checkpoint(model=test_runner.model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        self.model_dir,
        max_to_keep=None,
        step_counter=test_runner.global_step)
    test_controller = controller.Controller(
        trainer=test_runner,
        evaluator=test_runner,
        global_step=test_runner.global_step,
        checkpoint_manager=checkpoint_manager,
        enable_async_checkpointing=enable_async_checkpoint_saving,
        eval_summary_dir=os.path.join(self.model_dir, "summaries/eval"),
        steps_per_loop=2)
    test_controller.train_and_evaluate(
        train_steps=10, eval_steps=2, eval_interval=6)
    self.assertEqual(test_runner.global_step, 10)

    # Training summaries are not saved.
    self.assertEmpty(tf.io.gfile.glob(
        os.path.join(checkpoint_manager.directory, "events.*")))
    # Evaluation summaries are saved.
    self.assertNotEmpty(tf.io.gfile.glob(
        os.path.join(self.model_dir, "summaries/eval/events.*")))

  @parameterized.named_parameters(
      ("_sync_checkpoint_saving", False),
      ("_async_checkpoint_saving", True)
  )
  def test_restore_from_most_recent_checkpoint(
      self, enable_async_checkpoint_saving
  ):
    test_runner = TestRunner()
    checkpoint = tf.train.Checkpoint(model=test_runner.model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        self.model_dir,
        max_to_keep=None,
        step_counter=test_runner.global_step,
        checkpoint_interval=5)
    test_controller = controller.Controller(
        trainer=test_runner,
        global_step=test_runner.global_step,
        checkpoint_manager=checkpoint_manager,
        enable_async_checkpointing=enable_async_checkpoint_saving,
        eval_summary_dir=os.path.join(self.model_dir, "summaries/eval"),
        steps_per_loop=5)
    test_controller.train(20)
    self.assertLen(checkpoint_manager.checkpoints, 4)
    restored_path = test_controller.restore_checkpoint()
    self.assertEqual(restored_path, checkpoint_manager.checkpoints[-1])

  @parameterized.named_parameters(
      ("return_numpy_sync_checkpoint_saving", True, False),
      ("return_numpy_async_checkpoint_saving", True, True),
      ("return_tensor_sync_checkpoint_saving", False, False),
      ("return_tensor_async_checkpoint_saving", False, True),
  )
  def test_train_and_evaluate(
      self, return_numpy, enable_async_checkpoint_saving
  ):
    test_runner = TestRunner(return_numpy=return_numpy)

    checkpoint = tf.train.Checkpoint(
        model=test_runner.model, optimizer=test_runner.optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        self.model_dir,
        max_to_keep=None,
        step_counter=test_runner.global_step,
        checkpoint_interval=10)
    test_controller = controller.Controller(
        trainer=test_runner,
        evaluator=test_runner,
        global_step=test_runner.global_step,
        steps_per_loop=2,
        summary_dir=os.path.join(self.model_dir, "summaries/train"),
        checkpoint_manager=checkpoint_manager,
        enable_async_checkpointing=enable_async_checkpoint_saving,
        eval_summary_dir=os.path.join(self.model_dir, "summaries/eval"))
    test_controller.train_and_evaluate(
        train_steps=10, eval_steps=2, eval_interval=6)

    # Checkpoints are saved.
    self.assertNotEmpty(tf.io.gfile.glob(os.path.join(self.model_dir, "ckpt*")))

    # Loss and accuracy values should be written into summaries.
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(self.model_dir, "summaries/train")))
    self.assertNotEmpty(
        summaries_with_matching_keyword(
            "loss", os.path.join(self.model_dir, "summaries/train")))
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(self.model_dir, "summaries/eval")))
    self.assertNotEmpty(
        summaries_with_matching_keyword(
            "eval_loss", os.path.join(self.model_dir, "summaries/eval")))

  @parameterized.named_parameters(
      ("_sync_checkpoint_saving", False),
      ("_async_checkpoint_saving", True)
  )
  def test_train_only(self, enable_async_checkpoint_saving):
    test_runner = TestRunner()

    checkpoint = tf.train.Checkpoint(
        model=test_runner.model, optimizer=test_runner.optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        self.model_dir,
        max_to_keep=None,
        step_counter=test_runner.global_step,
        checkpoint_interval=10)
    test_controller = controller.Controller(
        trainer=test_runner,
        global_step=test_runner.global_step,
        steps_per_loop=2,
        summary_dir=os.path.join(self.model_dir, "summaries/train"),
        checkpoint_manager=checkpoint_manager,
        enable_async_checkpointing=enable_async_checkpoint_saving,
        eval_summary_dir=os.path.join(self.model_dir, "summaries/eval"),
    )
    test_controller.train(steps=10)

    # Checkpoints are saved.
    self.assertNotEmpty(tf.io.gfile.glob(os.path.join(self.model_dir, "ckpt*")))

    # Only train summaries are written.
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(self.model_dir, "summaries/train")))
    self.assertNotEmpty(
        summaries_with_matching_keyword(
            "loss", os.path.join(self.model_dir, "summaries/train")))
    self.assertFalse(
        tf.io.gfile.exists(os.path.join(self.model_dir, "summaries/eval")))

  def test_evaluate_only(self):
    test_runner = TestRunner()

    checkpoint = tf.train.Checkpoint(model=test_runner.model)
    checkpoint.save(os.path.join(self.model_dir, "ckpt"))
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        self.model_dir,
        max_to_keep=None,
        step_counter=test_runner.global_step)
    test_controller = controller.Controller(
        evaluator=test_runner,
        global_step=test_runner.global_step,
        checkpoint_manager=checkpoint_manager,
        summary_dir=os.path.join(self.model_dir, "summaries/train"),
        eval_summary_dir=os.path.join(self.model_dir, "summaries/eval"))
    eval_results = test_controller.evaluate(steps=2)

    # Only eval summaries are written
    self.assertFalse(
        tf.io.gfile.exists(os.path.join(self.model_dir, "summaries/train")))
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(self.model_dir, "summaries/eval")))
    self.assertNotEmpty(
        summaries_with_matching_keyword(
            "eval_loss", os.path.join(self.model_dir, "summaries/eval")))
    self.assertIn("eval_loss", eval_results)

    # Tests continuous eval with timeout and timeout_fn.
    done_file = os.path.join(self.model_dir, "summaries/eval/Done")

    def timeout_fn():
      with tf.io.gfile.GFile(done_file, "w") as f:
        f.write("DONE")
        return True

    test_controller = controller.Controller(
        evaluator=test_runner,
        global_step=test_runner.global_step,
        checkpoint_manager=checkpoint_manager,
        eval_summary_dir=os.path.join(self.model_dir, "summaries/eval"))
    test_controller.evaluate_continuously(
        timeout=1, timeout_fn=timeout_fn, steps=2)
    self.assertNotEmpty(tf.io.gfile.glob(done_file))

  def test_no_eval_steps(self):
    test_runner = TestRunner()

    checkpoint = tf.train.Checkpoint(model=test_runner.model)
    checkpoint.save(os.path.join(self.model_dir, "ckpt"))
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        self.model_dir,
        max_to_keep=None,
        step_counter=test_runner.global_step)
    test_controller = controller.Controller(
        evaluator=test_runner,
        global_step=test_runner.global_step,
        checkpoint_manager=checkpoint_manager)
    test_controller.evaluate()

  @parameterized.named_parameters(
      ("_sync_checkpoint_saving", False),
      ("_async_checkpoint_saving", True)
  )
  def test_already_trained_model(self, enable_async_checkpoint_saving):
    test_runner = TestRunner()
    test_runner.global_step.assign(10)

    checkpoint = tf.train.Checkpoint(
        model=test_runner.model, optimizer=test_runner.optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        self.model_dir,
        max_to_keep=None,
        step_counter=test_runner.global_step,
        checkpoint_interval=10)
    test_controller = controller.Controller(
        trainer=test_runner,
        global_step=test_runner.global_step,
        steps_per_loop=2,
        checkpoint_manager=checkpoint_manager,
        enable_async_checkpointing=enable_async_checkpoint_saving)
    # `global_step` is already `train_steps`.
    test_controller.train(steps=10)

  def test_summaries_inside_train_fn(self):
    test_runner = TestTrainerWithSummaries()

    checkpoint = tf.train.Checkpoint(
        model=test_runner.model, optimizer=test_runner.optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        self.model_dir,
        max_to_keep=None,
        step_counter=test_runner.global_step)
    test_controller = controller.Controller(
        trainer=test_runner,
        global_step=test_runner.global_step,
        steps_per_loop=2,
        summary_dir=os.path.join(self.model_dir, "summaries/train"),
        summary_interval=2,
        checkpoint_manager=checkpoint_manager
    )
    test_controller.train(steps=10)

    # Checkpoints are saved.
    self.assertEmpty(tf.io.gfile.glob(os.path.join(self.model_dir, "ckpt*")))

    # Only train summaries are written.
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(self.model_dir, "summaries/train")))
    self.assertNotEmpty(
        summaries_with_matching_keyword(
            "loss", os.path.join(self.model_dir, "summaries/train")))
    self.assertFalse(
        tf.io.gfile.exists(os.path.join(self.model_dir, "summaries/eval")))

  def test_train_and_evaluate_with_same_summary_dir(self):
    test_runner = TestRunner()

    checkpoint = tf.train.Checkpoint(
        model=test_runner.model, optimizer=test_runner.optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        self.model_dir,
        max_to_keep=None,
        step_counter=test_runner.global_step)
    test_controller = controller.Controller(
        trainer=test_runner,
        evaluator=test_runner,
        global_step=test_runner.global_step,
        steps_per_loop=2,
        summary_dir=os.path.join(self.model_dir, "summaries"),
        checkpoint_manager=checkpoint_manager,
        eval_summary_dir=os.path.join(self.model_dir, "summaries"))
    test_controller.train_and_evaluate(
        train_steps=10, eval_steps=2, eval_interval=6)

    # Loss and accuracy values should be written into summaries.
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(self.model_dir, "summaries")))
    self.assertNotEmpty(
        summaries_with_matching_keyword(
            "loss", os.path.join(self.model_dir, "summaries")))
    self.assertNotEmpty(
        summaries_with_matching_keyword(
            "eval_loss", os.path.join(self.model_dir, "summaries")))

  def test_early_stop_on_eval_loss(self):
    test_runner = TestRunner()

    class EarlyStopController(controller.Controller):
      """A subclass of Controller that supports early stopping."""

      def train_and_evaluate(self,
                             train_steps: int = None,
                             eval_steps: int = None,
                             eval_interval: int = None):
        while self.global_step.numpy() < train_steps:
          interval = min(train_steps - self.global_step.numpy(), eval_interval)
          num_steps = self.global_step.numpy() + interval
          self.train(steps=num_steps, checkpoint_at_completion=False)
          self._sync_on_async_checkpointing()
          self.evaluate(steps=eval_steps)
          # Early stop condition.
          if test_runner.eval_loss.result() < 0.1:
            logging.info(
                "Training early stopped as eval_loss %s is less than 0.1",
                test_runner.eval_loss.result())
            return

    checkpoint = tf.train.Checkpoint(
        model=test_runner.model, optimizer=test_runner.optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        self.model_dir,
        max_to_keep=None,
        step_counter=test_runner.global_step,
        checkpoint_interval=10)
    test_controller = EarlyStopController(
        trainer=test_runner,
        evaluator=test_runner,
        global_step=test_runner.global_step,
        steps_per_loop=2,
        checkpoint_manager=checkpoint_manager)
    test_controller.train_and_evaluate(
        train_steps=10, eval_steps=6, eval_interval=2)

    self.assertLess(test_runner.global_step, 10)

  def test_evaluate_with_loss_output(self):
    test_evaluator = TestEvaluator()

    checkpoint = tf.train.Checkpoint(model=test_evaluator.model)
    checkpoint.save(os.path.join(self.model_dir, "ckpt"))
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, self.model_dir, max_to_keep=None)
    test_controller = controller.Controller(
        evaluator=test_evaluator,
        global_step=tf.Variable(0, dtype=tf.int64),
        checkpoint_manager=checkpoint_manager,
        eval_summary_dir=os.path.join(self.model_dir, "summaries/eval"))
    test_controller.evaluate(steps=5)

    # Only eval summaries are written
    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(self.model_dir, "summaries/eval")))
    self.assertNotEmpty(
        summaries_with_matching_keyword(
            "eval_loss", os.path.join(self.model_dir, "summaries/eval")))

  def test_evaluate_with_no_output(self):
    test_controller = controller.Controller(
        evaluator=TestEvaluatorNoOutput(),
        global_step=tf.Variable(0, dtype=tf.int64),
        eval_summary_dir=os.path.join(self.model_dir, "summaries/eval"))
    self.assertSameElements(["steps_per_second"],
                            test_controller.evaluate(steps=5).keys())

  def test_train_and_evaluate_reset_datasets(self):
    test_runner = TestRunner()

    test_controller = controller.Controller(
        trainer=test_runner,
        evaluator=test_runner,
        global_step=test_runner.global_step,
        steps_per_loop=2)

    test_controller.train_and_evaluate(
        train_steps=10, eval_steps=2, eval_interval=6)

    train_dataset = (
        test_runner.strategy.distribute_datasets_from_function(dataset_fn))
    eval_dataset = (
        test_runner.strategy.distribute_datasets_from_function(dataset_fn))
    test_runner.train_dataset = train_dataset
    test_runner.eval_dataset = eval_dataset

    test_controller.train_and_evaluate(
        train_steps=10, eval_steps=2, eval_interval=6)

  @parameterized.named_parameters(
      ("_sync_checkpoint_saving", False),
      ("_async_checkpoint_saving", True)
  )
  def test_eval_and_checkpoint_interval(self, enable_async_checkpoint_saving):
    test_runner = TestRunner()

    checkpoint = tf.train.Checkpoint(
        model=test_runner.model, optimizer=test_runner.optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        self.model_dir,
        max_to_keep=None,
        step_counter=test_runner.global_step,
        checkpoint_interval=5)
    test_controller = controller.Controller(
        trainer=test_runner,
        evaluator=test_runner,
        global_step=test_runner.global_step,
        steps_per_loop=10,
        checkpoint_manager=checkpoint_manager,
        enable_async_checkpointing=enable_async_checkpoint_saving,
        summary_dir=self.model_dir)
    test_controller.train_and_evaluate(
        train_steps=10, eval_steps=2, eval_interval=5)

    # Expect 3 checkpoints to be saved at step: 5, 10.
    self.assertLen(
        tf.io.gfile.glob(os.path.join(self.model_dir, "ckpt-*.data*")), 2)
    # Expect evaluation is performed 2 times at step: 5, 10.
    self.assertLen(
        summaries_with_matching_keyword("eval_loss", self.model_dir), 2)

  @parameterized.named_parameters(("DefaultSummary", False),
                                  ("InjectSummary", True))
  def test_evaluate_with_nested_summaries(self, inject_summary_manager):
    test_evaluator = TestEvaluatorWithNestedSummary()
    if inject_summary_manager:
      summary_manager = orbit.utils.SummaryManager(
          self.model_dir,
          tf.summary.scalar,
          global_step=tf.Variable(0, dtype=tf.int64))
    else:
      summary_manager = None
    test_controller = controller.Controller(
        evaluator=test_evaluator,
        global_step=tf.Variable(0, dtype=tf.int64),
        eval_summary_dir=self.model_dir,
        summary_manager=summary_manager)
    test_controller.evaluate(steps=5)

    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(self.model_dir, "dataset")))
    self.assertNotEmpty(
        summaries_with_matching_keyword(
            "loss", os.path.join(self.model_dir, "dataset")))
    self.assertNotEmpty(
        summaries_with_matching_keyword(
            "accuracy", os.path.join(self.model_dir, "dataset")))

    self.assertNotEmpty(
        tf.io.gfile.listdir(os.path.join(self.model_dir, "dataset2")))
    self.assertNotEmpty(
        summaries_with_matching_keyword(
            "loss", os.path.join(self.model_dir, "dataset2")))
    self.assertNotEmpty(
        summaries_with_matching_keyword(
            "accuracy", os.path.join(self.model_dir, "dataset2")))

  def test_actions(self):
    test_runner = TestRunner()
    checkpoint = tf.train.Checkpoint(
        model=test_runner.model, optimizer=test_runner.optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        self.model_dir,
        max_to_keep=None,
        step_counter=test_runner.global_step,
        checkpoint_interval=10)

    class OutputRecorderAction:
      """Simple `Action` that just saves the outputs passed to `__call__`."""

      def __init__(self):
        self.outputs = []

      def __call__(self, output):
        self.outputs.append(output)

    train_output_recorder = OutputRecorderAction()
    eval_output_recorder = OutputRecorderAction()

    test_controller = controller.Controller(
        trainer=test_runner,
        evaluator=test_runner,
        train_actions=[train_output_recorder],
        eval_actions=[eval_output_recorder],
        global_step=test_runner.global_step,
        steps_per_loop=2,
        summary_dir=os.path.join(self.model_dir, "summaries/train"),
        checkpoint_manager=checkpoint_manager,
        eval_summary_dir=os.path.join(self.model_dir, "summaries/eval"))
    test_controller.train_and_evaluate(
        train_steps=10, eval_steps=2, eval_interval=6)

    self.assertLen(train_output_recorder.outputs, 5)
    for output in train_output_recorder.outputs:
      self.assertIn("loss", output)
      self.assertGreaterEqual(output["loss"], 0)

    self.assertLen(eval_output_recorder.outputs, 2)
    for output in eval_output_recorder.outputs:
      self.assertIn("eval_loss", output)
      self.assertGreaterEqual(output["eval_loss"], 0)

  def test_step_per_loop_callable(self):
    test_runner = TestRunner()

    checkpoint = tf.train.Checkpoint(
        model=test_runner.model, optimizer=test_runner.optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        self.model_dir,
        max_to_keep=None,
        step_counter=test_runner.global_step,
        checkpoint_interval=10)

    def steps_per_loop_fn(global_step):
      if global_step > 4:
        return 4
      return 2

    test_controller = controller.Controller(
        trainer=test_runner,
        global_step=test_runner.global_step,
        steps_per_loop=steps_per_loop_fn,
        checkpoint_manager=checkpoint_manager
    )
    test_controller.train(steps=10)
    self.assertEqual(test_runner.global_step, 10)


if __name__ == "__main__":
  tf.test.main()

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
"""Tests for official.modeling.training.model_training_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.modeling import model_training_utils


def eager_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.tpu_strategy,
          strategy_combinations.one_device_strategy_gpu,
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_two_gpus,
      ],
      mode='eager',
  )


def eager_gpu_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.one_device_strategy_gpu,
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_two_gpus,
      ],
      mode='eager',
  )


def create_fake_data_input_fn(batch_size, features_shape, num_classes):
  """Creates a dummy input function with the given feature and label shapes.

  Args:
    batch_size: integer.
    features_shape: list[int]. Feature shape for an individual example.
    num_classes: integer. Number of labels.

  Returns:
    An input function that is usable in the executor.
  """

  def _input_fn():
    """An input function for generating fake data."""
    features = np.random.rand(64, *features_shape)
    labels = np.random.randint(2, size=[64, num_classes])
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    def _assign_dtype(features, labels):
      features = tf.cast(features, tf.float32)
      labels = tf.cast(labels, tf.float32)
      return features, labels

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.map(_assign_dtype)
    dataset = dataset.shuffle(64).repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=64)
    return dataset

  return _input_fn


def create_model_fn(input_shape, num_classes, use_float16=False):

  def _model_fn():
    """A one-layer softmax model suitable for testing."""
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(num_classes, activation='relu')(input_layer)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    sub_model = tf.keras.models.Model(input_layer, x, name='sub_model')
    model = tf.keras.models.Model(input_layer, output_layer, name='model')
    model.add_metric(
        tf.reduce_mean(input_layer), name='mean_input', aggregation='mean')
    model.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    if use_float16:
      model.optimizer = (
          tf.keras.mixed_precision.experimental.LossScaleOptimizer(
              model.optimizer, loss_scale='dynamic'))
    return model, sub_model

  return _model_fn


def metric_fn():
  """Gets a tf.keras metric object."""
  return tf.keras.metrics.CategoricalAccuracy(name='accuracy', dtype=tf.float32)


def summaries_with_matching_keyword(keyword, summary_dir):
  """Yields summary protos matching given keyword from event file."""
  event_paths = tf.io.gfile.glob(os.path.join(summary_dir, 'events*'))
  for event in tf.compat.v1.train.summary_iterator(event_paths[-1]):
    if event.summary is not None:
      for value in event.summary.value:
        if keyword in value.tag:
          tf.compat.v1.logging.error(event)
          yield event.summary


def check_eventfile_for_keyword(keyword, summary_dir):
  """Checks event files for the keyword."""
  return any(summaries_with_matching_keyword(keyword, summary_dir))


class ModelTrainingUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ModelTrainingUtilsTest, self).setUp()
    self._input_fn = create_fake_data_input_fn(
        batch_size=8, features_shape=[128], num_classes=3)
    self._model_fn = create_model_fn(input_shape=[128], num_classes=3)

  def run_training(self, distribution, model_dir, steps_per_loop, run_eagerly):
    model_training_utils.run_customized_training_loop(
        strategy=distribution,
        model_fn=self._model_fn,
        loss_fn=tf.keras.losses.categorical_crossentropy,
        model_dir=model_dir,
        steps_per_epoch=20,
        steps_per_loop=steps_per_loop,
        epochs=2,
        train_input_fn=self._input_fn,
        eval_input_fn=self._input_fn,
        eval_steps=10,
        init_checkpoint=None,
        metric_fn=metric_fn,
        custom_callbacks=None,
        run_eagerly=run_eagerly)

  @combinations.generate(eager_strategy_combinations())
  def test_train_eager_single_step(self, distribution):
    model_dir = self.get_temp_dir()
    if isinstance(distribution, tf.distribute.experimental.TPUStrategy):
      with self.assertRaises(ValueError):
        self.run_training(
            distribution, model_dir, steps_per_loop=1, run_eagerly=True)
    else:
      self.run_training(
          distribution, model_dir, steps_per_loop=1, run_eagerly=True)

  @combinations.generate(eager_gpu_strategy_combinations())
  def test_train_eager_mixed_precision(self, distribution):
    model_dir = self.get_temp_dir()
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    self._model_fn = create_model_fn(
        input_shape=[128], num_classes=3, use_float16=True)
    self.run_training(
        distribution, model_dir, steps_per_loop=1, run_eagerly=True)

  @combinations.generate(eager_strategy_combinations())
  def test_train_check_artifacts(self, distribution):
    model_dir = self.get_temp_dir()
    self.run_training(
        distribution, model_dir, steps_per_loop=10, run_eagerly=False)

    # Two checkpoints should be saved after two epochs.
    self.assertNotEmpty(tf.io.gfile.glob(os.path.join(model_dir, 'ctl_step_*')))
    self.assertNotEmpty(
        tf.io.gfile.glob(
            os.path.join(model_dir, 'summaries/training_summary*')))

    # Loss and accuracy values should be written into summaries.
    self.assertTrue(
        check_eventfile_for_keyword('loss',
                                    os.path.join(model_dir, 'summaries/train')))
    self.assertTrue(
        check_eventfile_for_keyword('accuracy',
                                    os.path.join(model_dir, 'summaries/train')))
    self.assertTrue(
        check_eventfile_for_keyword('mean_input',
                                    os.path.join(model_dir, 'summaries/train')))
    self.assertTrue(
        check_eventfile_for_keyword('accuracy',
                                    os.path.join(model_dir, 'summaries/eval')))
    self.assertTrue(
        check_eventfile_for_keyword('mean_input',
                                    os.path.join(model_dir, 'summaries/eval')))


if __name__ == '__main__':
  assert tf.version.VERSION.startswith('2.')
  tf.test.main()

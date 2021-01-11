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

import os

from absl import logging
from absl.testing import flagsaver
from absl.testing import parameterized
from absl.testing.absltest import mock
import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.nlp.bert import common_flags
from official.nlp.bert import model_training_utils


common_flags.define_common_bert_flags()


def eager_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.cloud_tpu_strategy,
          strategy_combinations.one_device_strategy_gpu,
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_two_gpus,
      ],)


def eager_gpu_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.one_device_strategy_gpu,
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_two_gpus,
      ],)


def create_fake_data_input_fn(batch_size, features_shape, num_classes):
  """Creates a dummy input function with the given feature and label shapes.

  Args:
    batch_size: integer.
    features_shape: list[int]. Feature shape for an individual example.
    num_classes: integer. Number of labels.

  Returns:
    An input function that is usable in the executor.
  """

  def _dataset_fn(input_context=None):
    """An input function for generating fake data."""
    local_batch_size = input_context.get_per_replica_batch_size(batch_size)
    features = np.random.rand(64, *features_shape)
    labels = np.random.randint(2, size=[64, num_classes])
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)

    def _assign_dtype(features, labels):
      features = tf.cast(features, tf.float32)
      labels = tf.cast(labels, tf.float32)
      return features, labels

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.map(_assign_dtype)
    dataset = dataset.shuffle(64).repeat()
    dataset = dataset.batch(local_batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=64)
    return dataset

  return _dataset_fn


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
      model.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
          model.optimizer)
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
          logging.error(event)
          yield event.summary


def check_eventfile_for_keyword(keyword, summary_dir):
  """Checks event files for the keyword."""
  return any(summaries_with_matching_keyword(keyword, summary_dir))


class RecordingCallback(tf.keras.callbacks.Callback):

  def __init__(self):
    self.batch_begin = []  # (batch, logs)
    self.batch_end = []  # (batch, logs)
    self.epoch_begin = []  # (epoch, logs)
    self.epoch_end = []  # (epoch, logs)

  def on_batch_begin(self, batch, logs=None):
    self.batch_begin.append((batch, logs))

  def on_batch_end(self, batch, logs=None):
    self.batch_end.append((batch, logs))

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch_begin.append((epoch, logs))

  def on_epoch_end(self, epoch, logs=None):
    self.epoch_end.append((epoch, logs))


class ModelTrainingUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ModelTrainingUtilsTest, self).setUp()
    self._model_fn = create_model_fn(input_shape=[128], num_classes=3)

  @flagsaver.flagsaver
  def run_training(self, strategy, model_dir, steps_per_loop, run_eagerly):
    input_fn = create_fake_data_input_fn(
        batch_size=8, features_shape=[128], num_classes=3)
    model_training_utils.run_customized_training_loop(
        strategy=strategy,
        model_fn=self._model_fn,
        loss_fn=tf.keras.losses.categorical_crossentropy,
        model_dir=model_dir,
        steps_per_epoch=20,
        steps_per_loop=steps_per_loop,
        epochs=2,
        train_input_fn=input_fn,
        eval_input_fn=input_fn,
        eval_steps=10,
        init_checkpoint=None,
        sub_model_export_name='my_submodel_name',
        metric_fn=metric_fn,
        custom_callbacks=None,
        run_eagerly=run_eagerly)

  @combinations.generate(eager_strategy_combinations())
  def test_train_eager_single_step(self, distribution):
    model_dir = self.create_tempdir().full_path
    if isinstance(
        distribution,
        (tf.distribute.TPUStrategy, tf.distribute.experimental.TPUStrategy)):
      with self.assertRaises(ValueError):
        self.run_training(
            distribution, model_dir, steps_per_loop=1, run_eagerly=True)
    else:
      self.run_training(
          distribution, model_dir, steps_per_loop=1, run_eagerly=True)

  @combinations.generate(eager_gpu_strategy_combinations())
  def test_train_eager_mixed_precision(self, distribution):
    model_dir = self.create_tempdir().full_path
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    self._model_fn = create_model_fn(
        input_shape=[128], num_classes=3, use_float16=True)
    self.run_training(
        distribution, model_dir, steps_per_loop=1, run_eagerly=True)

  @combinations.generate(eager_strategy_combinations())
  def test_train_check_artifacts(self, distribution):
    model_dir = self.create_tempdir().full_path
    self.run_training(
        distribution, model_dir, steps_per_loop=10, run_eagerly=False)

    # Two checkpoints should be saved after two epochs.
    files = map(os.path.basename,
                tf.io.gfile.glob(os.path.join(model_dir, 'ctl_step_*index')))
    self.assertCountEqual(
        ['ctl_step_20.ckpt-1.index', 'ctl_step_40.ckpt-2.index'], files)

    # Three submodel checkpoints should be saved after two epochs (one after
    # each epoch plus one final).
    files = map(
        os.path.basename,
        tf.io.gfile.glob(os.path.join(model_dir, 'my_submodel_name*index')))
    self.assertCountEqual([
        'my_submodel_name.ckpt-3.index',
        'my_submodel_name_step_20.ckpt-1.index',
        'my_submodel_name_step_40.ckpt-2.index'
    ], files)

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

  @combinations.generate(eager_strategy_combinations())
  def test_train_check_callbacks(self, distribution):
    model_dir = self.create_tempdir().full_path
    callback = RecordingCallback()
    callbacks = [callback]
    input_fn = create_fake_data_input_fn(
        batch_size=8, features_shape=[128], num_classes=3)
    model_training_utils.run_customized_training_loop(
        strategy=distribution,
        model_fn=self._model_fn,
        loss_fn=tf.keras.losses.categorical_crossentropy,
        model_dir=model_dir,
        steps_per_epoch=20,
        num_eval_per_epoch=4,
        steps_per_loop=10,
        epochs=2,
        train_input_fn=input_fn,
        eval_input_fn=input_fn,
        eval_steps=10,
        init_checkpoint=None,
        metric_fn=metric_fn,
        custom_callbacks=callbacks,
        run_eagerly=False)
    self.assertEqual(callback.epoch_begin, [(1, {}), (2, {})])
    epoch_ends, epoch_end_infos = zip(*callback.epoch_end)
    self.assertEqual(list(epoch_ends), [1, 2, 2])
    for info in epoch_end_infos:
      self.assertIn('accuracy', info)

    self.assertEqual(callback.batch_begin, [(0, {}), (5, {}), (10, {}),
                                            (15, {}), (20, {}), (25, {}),
                                            (30, {}), (35, {})])
    batch_ends, batch_end_infos = zip(*callback.batch_end)
    self.assertEqual(list(batch_ends), [4, 9, 14, 19, 24, 29, 34, 39])
    for info in batch_end_infos:
      self.assertIn('loss', info)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.one_device_strategy_gpu,
          ],))
  def test_train_check_artifacts_non_chief(self, distribution):
    # We shouldn't export artifacts on non-chief workers. Since there's no easy
    # way to test with real MultiWorkerMirroredStrategy, we patch the strategy
    # to make it as if it's MultiWorkerMirroredStrategy on non-chief workers.
    extended = distribution.extended
    with mock.patch.object(extended.__class__, 'should_checkpoint',
                           new_callable=mock.PropertyMock, return_value=False), \
         mock.patch.object(extended.__class__, 'should_save_summary',
                           new_callable=mock.PropertyMock, return_value=False):
      model_dir = self.create_tempdir().full_path
      self.run_training(
          distribution, model_dir, steps_per_loop=10, run_eagerly=False)
      self.assertEmpty(tf.io.gfile.listdir(model_dir))


if __name__ == '__main__':
  tf.test.main()

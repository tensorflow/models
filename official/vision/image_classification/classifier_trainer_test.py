# Lint as: python3
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Unit tests for the classifier trainer models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import json

import os
import sys

from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Tuple

from absl import flags
from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.utils.flags import core as flags_core
from official.vision.image_classification import classifier_trainer
from official.vision.image_classification import dataset_factory
from official.vision.image_classification import test_utils
from official.vision.image_classification.configs import base_configs

classifier_trainer.define_classifier_flags()


def distribution_strategy_combinations() -> Iterable[Tuple[Any, ...]]:
  """Returns the combinations of end-to-end tests to run."""
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.tpu_strategy,
          strategy_combinations.one_device_strategy_gpu,
          strategy_combinations.mirrored_strategy_with_two_gpus,
      ],
      model=[
          'efficientnet',
          'resnet',
      ],
      mode='eager',
      dataset=[
          'imagenet',
      ],
  )


def get_params_override(params_override: Mapping[str, Any]) -> str:
  """Converts params_override dict to string command."""
  return '--params_override=' + json.dumps(params_override)


def basic_params_override(dtype: str = 'float32') -> MutableMapping[str, Any]:
  """Returns a basic parameter configuration for testing."""
  return {
      'train_dataset': {
          'builder': 'synthetic',
          'use_per_replica_batch_size': True,
          'batch_size': 1,
          'image_size': 224,
          'dtype': dtype,
      },
      'validation_dataset': {
          'builder': 'synthetic',
          'batch_size': 1,
          'use_per_replica_batch_size': True,
          'image_size': 224,
          'dtype': dtype,
      },
      'train': {
          'steps': 1,
          'epochs': 1,
          'callbacks': {
              'enable_checkpoint_and_export': True,
              'enable_tensorboard': False,
          },
      },
      'evaluation': {
          'steps': 1,
      },
  }


def get_trivial_model(num_classes: int) -> tf.keras.Model:
  """Creates and compiles trivial model for ImageNet dataset."""
  model = test_utils.trivial_model(num_classes=num_classes)
  lr = 0.01
  optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
  loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
  model.compile(optimizer=optimizer,
                loss=loss_obj,
                run_eagerly=True)
  return model


def get_trivial_data() -> tf.data.Dataset:
  """Gets trivial data in the ImageNet size."""
  def generate_data(_) -> tf.data.Dataset:
    image = tf.zeros(shape=(224, 224, 3), dtype=tf.float32)
    label = tf.zeros([1], dtype=tf.int32)
    return image, label

  dataset = tf.data.Dataset.range(1)
  dataset = dataset.repeat()
  dataset = dataset.map(generate_data,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(buffer_size=1).batch(1)
  return dataset


def run_end_to_end(main: Callable[[Any], None],
                   extra_flags: Optional[Iterable[str]] = None,
                   model_dir: Optional[str] = None):
  """Runs the classifier trainer end-to-end."""
  extra_flags = [] if extra_flags is None else extra_flags
  args = [sys.argv[0], '--model_dir', model_dir] + extra_flags
  flags_core.parse_flags(argv=args)
  main(flags.FLAGS)


class ClassifierTest(tf.test.TestCase, parameterized.TestCase):
  """Unit tests for Keras models."""
  _tempdir = None

  @classmethod
  def setUpClass(cls):  # pylint: disable=invalid-name
    super(ClassifierTest, cls).setUpClass()

  def tearDown(self):
    super(ClassifierTest, self).tearDown()
    tf.io.gfile.rmtree(self.get_temp_dir())

  @combinations.generate(distribution_strategy_combinations())
  def test_end_to_end_train_and_eval(self, distribution, model, dataset):
    """Test train_and_eval and export for Keras classifier models."""
    # Some parameters are not defined as flags (e.g. cannot run
    # classifier_train.py --batch_size=...) by design, so use
    # "--params_override=..." instead
    model_dir = self.get_temp_dir()
    base_flags = [
        '--data_dir=not_used',
        '--model_type=' + model,
        '--dataset=' + dataset,
    ]
    train_and_eval_flags = base_flags + [
        get_params_override(basic_params_override()),
        '--mode=train_and_eval',
    ]

    run = functools.partial(classifier_trainer.run,
                            strategy_override=distribution)
    run_end_to_end(main=run,
                   extra_flags=train_and_eval_flags,
                   model_dir=model_dir)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.one_device_strategy_gpu,
          ],
          model=[
              'efficientnet',
              'resnet',
          ],
          mode='eager',
          dataset='imagenet',
          dtype='float16',
      ))
  def test_gpu_train(self, distribution, model, dataset, dtype):
    """Test train_and_eval and export for Keras classifier models."""
    # Some parameters are not defined as flags (e.g. cannot run
    # classifier_train.py --batch_size=...) by design, so use
    # "--params_override=..." instead
    model_dir = self.get_temp_dir()
    base_flags = [
        '--data_dir=not_used',
        '--model_type=' + model,
        '--dataset=' + dataset,
    ]
    train_and_eval_flags = base_flags + [
        get_params_override(basic_params_override(dtype)),
        '--mode=train_and_eval',
    ]

    export_params = basic_params_override()
    export_path = os.path.join(model_dir, 'export')
    export_params['export'] = {}
    export_params['export']['destination'] = export_path
    export_flags = base_flags + [
        '--mode=export_only',
        get_params_override(export_params)
    ]

    run = functools.partial(classifier_trainer.run,
                            strategy_override=distribution)
    run_end_to_end(main=run,
                   extra_flags=train_and_eval_flags,
                   model_dir=model_dir)
    run_end_to_end(main=run,
                   extra_flags=export_flags,
                   model_dir=model_dir)
    self.assertTrue(os.path.exists(export_path))

  @combinations.generate(
      combinations.combine(
      distribution=[
          strategy_combinations.tpu_strategy,
      ],
      model=[
          'efficientnet',
          'resnet',
      ],
      mode='eager',
      dataset='imagenet',
      dtype='bfloat16',
  ))
  def test_tpu_train(self, distribution, model, dataset, dtype):
    """Test train_and_eval and export for Keras classifier models."""
    # Some parameters are not defined as flags (e.g. cannot run
    # classifier_train.py --batch_size=...) by design, so use
    # "--params_override=..." instead
    model_dir = self.get_temp_dir()
    base_flags = [
        '--data_dir=not_used',
        '--model_type=' + model,
        '--dataset=' + dataset,
    ]
    train_and_eval_flags = base_flags + [
        get_params_override(basic_params_override(dtype)),
        '--mode=train_and_eval',
    ]

    run = functools.partial(classifier_trainer.run,
                            strategy_override=distribution)
    run_end_to_end(main=run,
                   extra_flags=train_and_eval_flags,
                   model_dir=model_dir)

  @combinations.generate(distribution_strategy_combinations())
  def test_end_to_end_invalid_mode(self, distribution, model, dataset):
    """Test the Keras EfficientNet model with `strategy`."""
    model_dir = self.get_temp_dir()
    extra_flags = [
        '--data_dir=not_used',
        '--mode=invalid_mode',
        '--model_type=' + model,
        '--dataset=' + dataset,
        get_params_override(basic_params_override()),
    ]

    run = functools.partial(classifier_trainer.run,
                            strategy_override=distribution)
    with self.assertRaises(ValueError):
      run_end_to_end(main=run, extra_flags=extra_flags, model_dir=model_dir)


class UtilTests(parameterized.TestCase, tf.test.TestCase):
  """Tests for individual utility functions within classifier_trainer.py."""

  @parameterized.named_parameters(
      ('efficientnet-b0', 'efficientnet', 'efficientnet-b0', 224),
      ('efficientnet-b1', 'efficientnet', 'efficientnet-b1', 240),
      ('efficientnet-b2', 'efficientnet', 'efficientnet-b2', 260),
      ('efficientnet-b3', 'efficientnet', 'efficientnet-b3', 300),
      ('efficientnet-b4', 'efficientnet', 'efficientnet-b4', 380),
      ('efficientnet-b5', 'efficientnet', 'efficientnet-b5', 456),
      ('efficientnet-b6', 'efficientnet', 'efficientnet-b6', 528),
      ('efficientnet-b7', 'efficientnet', 'efficientnet-b7', 600),
      ('resnet', 'resnet', '', None),
  )
  def test_get_model_size(self, model, model_name, expected):
    config = base_configs.ExperimentConfig(
        model_name=model,
        model=base_configs.ModelConfig(
            model_params={
                'model_name': model_name,
            },
        )
    )
    size = classifier_trainer.get_image_size_from_model(config)
    self.assertEqual(size, expected)

  @parameterized.named_parameters(
      ('dynamic', 'dynamic', None, 'dynamic'),
      ('scalar', 128., None, 128.),
      ('float32', None, 'float32', 1),
      ('float16', None, 'float16', 128),
  )
  def test_get_loss_scale(self, loss_scale, dtype, expected):
    config = base_configs.ExperimentConfig(
        runtime=base_configs.RuntimeConfig(
            loss_scale=loss_scale),
        train_dataset=dataset_factory.DatasetConfig(dtype=dtype))
    ls = classifier_trainer.get_loss_scale(config, fp16_default=128)
    self.assertEqual(ls, expected)

  @parameterized.named_parameters(
      ('float16', 'float16'),
      ('bfloat16', 'bfloat16')
  )
  def test_initialize(self, dtype):
    config = base_configs.ExperimentConfig(
        runtime=base_configs.RuntimeConfig(
            run_eagerly=False,
            enable_xla=False,
            per_gpu_thread_count=1,
            gpu_thread_mode='gpu_private',
            num_gpus=1,
            dataset_num_private_threads=1,
        ),
        train_dataset=dataset_factory.DatasetConfig(dtype=dtype),
        model=base_configs.ModelConfig(),
    )

    class EmptyClass:
      pass
    fake_ds_builder = EmptyClass()
    fake_ds_builder.dtype = dtype
    fake_ds_builder.config = EmptyClass()
    classifier_trainer.initialize(config, fake_ds_builder)

  def test_resume_from_checkpoint(self):
    """Tests functionality for resuming from checkpoint."""
    # Set the keras policy
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

    # Get the model, datasets, and compile it.
    model = get_trivial_model(10)

    # Create the checkpoint
    model_dir = self.get_temp_dir()
    train_epochs = 1
    train_steps = 10
    ds = get_trivial_data()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'model.ckpt-{epoch:04d}'),
            save_weights_only=True)
    ]
    model.fit(
        ds,
        callbacks=callbacks,
        epochs=train_epochs,
        steps_per_epoch=train_steps)

    # Test load from checkpoint
    clean_model = get_trivial_model(10)
    weights_before_load = copy.deepcopy(clean_model.get_weights())
    initial_epoch = classifier_trainer.resume_from_checkpoint(
        model=clean_model,
        model_dir=model_dir,
        train_steps=train_steps)
    self.assertEqual(initial_epoch, 1)
    self.assertNotAllClose(weights_before_load, clean_model.get_weights())

    tf.io.gfile.rmtree(model_dir)

  def test_serialize_config(self):
    """Tests functionality for serializing data."""
    config = base_configs.ExperimentConfig()
    model_dir = self.get_temp_dir()
    classifier_trainer.serialize_config(params=config, model_dir=model_dir)
    saved_params_path = os.path.join(model_dir, 'params.yaml')
    self.assertTrue(os.path.exists(saved_params_path))
    tf.io.gfile.rmtree(model_dir)

if __name__ == '__main__':
  tf.test.main()

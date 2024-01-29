# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Unit tests for the classifier trainer models."""

import functools
import json

import os
import sys

from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Tuple

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.legacy.image_classification import classifier_trainer
from official.utils.flags import core as flags_core


classifier_trainer.define_classifier_flags()


def distribution_strategy_combinations() -> Iterable[Tuple[Any, ...]]:
  """Returns the combinations of end-to-end tests to run."""
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.cloud_tpu_strategy,
          strategy_combinations.one_device_strategy_gpu,
          strategy_combinations.mirrored_strategy_with_two_gpus,
      ],
      model=[
          'efficientnet',
          'resnet',
          'vgg',
      ],
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


@flagsaver.flagsaver
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
    model_dir = self.create_tempdir().full_path
    base_flags = [
        '--data_dir=not_used',
        '--model_type=' + model,
        '--dataset=' + dataset,
    ]
    train_and_eval_flags = base_flags + [
        get_params_override(basic_params_override()),
        '--mode=train_and_eval',
    ]

    run = functools.partial(
        classifier_trainer.run, strategy_override=distribution)
    run_end_to_end(
        main=run, extra_flags=train_and_eval_flags, model_dir=model_dir)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.one_device_strategy_gpu,
          ],
          model=[
              'efficientnet',
              'resnet',
              'vgg',
          ],
          dataset='imagenet',
          dtype='float16',
      ))
  def test_gpu_train(self, distribution, model, dataset, dtype):
    """Test train_and_eval and export for Keras classifier models."""
    # Some parameters are not defined as flags (e.g. cannot run
    # classifier_train.py --batch_size=...) by design, so use
    # "--params_override=..." instead
    model_dir = self.create_tempdir().full_path
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

    run = functools.partial(
        classifier_trainer.run, strategy_override=distribution)
    run_end_to_end(
        main=run, extra_flags=train_and_eval_flags, model_dir=model_dir)
    run_end_to_end(main=run, extra_flags=export_flags, model_dir=model_dir)
    self.assertTrue(os.path.exists(export_path))

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.cloud_tpu_strategy,
          ],
          model=[
              'efficientnet',
              'resnet',
              'vgg',
          ],
          dataset='imagenet',
          dtype='bfloat16',
      ))
  def test_tpu_train(self, distribution, model, dataset, dtype):
    """Test train_and_eval and export for Keras classifier models."""
    # Some parameters are not defined as flags (e.g. cannot run
    # classifier_train.py --batch_size=...) by design, so use
    # "--params_override=..." instead
    model_dir = self.create_tempdir().full_path
    base_flags = [
        '--data_dir=not_used',
        '--model_type=' + model,
        '--dataset=' + dataset,
    ]
    train_and_eval_flags = base_flags + [
        get_params_override(basic_params_override(dtype)),
        '--mode=train_and_eval',
    ]

    run = functools.partial(
        classifier_trainer.run, strategy_override=distribution)
    run_end_to_end(
        main=run, extra_flags=train_and_eval_flags, model_dir=model_dir)

  @combinations.generate(distribution_strategy_combinations())
  def test_end_to_end_invalid_mode(self, distribution, model, dataset):
    """Test the Keras EfficientNet model with `strategy`."""
    model_dir = self.create_tempdir().full_path
    extra_flags = [
        '--data_dir=not_used',
        '--mode=invalid_mode',
        '--model_type=' + model,
        '--dataset=' + dataset,
        get_params_override(basic_params_override()),
    ]

    run = functools.partial(
        classifier_trainer.run, strategy_override=distribution)
    with self.assertRaises(ValueError):
      run_end_to_end(main=run, extra_flags=extra_flags, model_dir=model_dir)


if __name__ == '__main__':
  tf.test.main()

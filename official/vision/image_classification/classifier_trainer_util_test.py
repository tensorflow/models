# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

# Lint as: python3
"""Unit tests for the classifier trainer models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os

from absl.testing import parameterized
import tensorflow as tf

from official.vision.image_classification import classifier_trainer
from official.vision.image_classification import dataset_factory
from official.vision.image_classification import test_utils
from official.vision.image_classification.configs import base_configs


def get_trivial_model(num_classes: int) -> tf.keras.Model:
  """Creates and compiles trivial model for ImageNet dataset."""
  model = test_utils.trivial_model(num_classes=num_classes)
  lr = 0.01
  optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
  loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
  model.compile(optimizer=optimizer, loss=loss_obj, run_eagerly=True)
  return model


def get_trivial_data() -> tf.data.Dataset:
  """Gets trivial data in the ImageNet size."""

  def generate_data(_) -> tf.data.Dataset:
    image = tf.zeros(shape=(224, 224, 3), dtype=tf.float32)
    label = tf.zeros([1], dtype=tf.int32)
    return image, label

  dataset = tf.data.Dataset.range(1)
  dataset = dataset.repeat()
  dataset = dataset.map(
      generate_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(buffer_size=1).batch(1)
  return dataset


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
            },))
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
        runtime=base_configs.RuntimeConfig(loss_scale=loss_scale),
        train_dataset=dataset_factory.DatasetConfig(dtype=dtype))
    ls = classifier_trainer.get_loss_scale(config, fp16_default=128)
    self.assertEqual(ls, expected)

  @parameterized.named_parameters(('float16', 'float16'),
                                  ('bfloat16', 'bfloat16'))
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
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    # Get the model, datasets, and compile it.
    model = get_trivial_model(10)

    # Create the checkpoint
    model_dir = self.create_tempdir().full_path
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
        model=clean_model, model_dir=model_dir, train_steps=train_steps)
    self.assertEqual(initial_epoch, 1)
    self.assertNotAllClose(weights_before_load, clean_model.get_weights())

    tf.io.gfile.rmtree(model_dir)

  def test_serialize_config(self):
    """Tests functionality for serializing data."""
    config = base_configs.ExperimentConfig()
    model_dir = self.create_tempdir().full_path
    classifier_trainer.serialize_config(params=config, model_dir=model_dir)
    saved_params_path = os.path.join(model_dir, 'params.yaml')
    self.assertTrue(os.path.exists(saved_params_path))
    tf.io.gfile.rmtree(model_dir)


if __name__ == '__main__':
  tf.test.main()

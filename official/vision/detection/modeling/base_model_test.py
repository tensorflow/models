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
"""Tests for base_model.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf

from official.vision.detection.dataloader import mode_keys
from official.vision.detection.modeling import base_model
from official.modeling.hyperparams import params_dict


class DummyModel(base_model.Model):

  def build_model(self):
    input_shape = [1]
    input_layer = tf.keras.layers.Input(shape=input_shape)
    outputs = self.model_outputs(inputs=input_layer, mode=None)

    model = tf.keras.models.Model(
        inputs=input_layer, outputs=outputs, name='dummy')
    model.optimizer = self.build_optimizer()

    return model

  def build_loss_fn(self):
    return tf.keras.losses.MeanSquaredError()

  def build_outputs(self, features, mode):
    return tf.keras.layers.Dense(1)(features)


class BaseModelTest(tf.test.TestCase):

  def setUp(self):
    super(BaseModelTest, self).setUp()
    self._model_dir = os.path.join(self.get_temp_dir(),
                                   'model_dir')

  def testBaseModelTrainAndEval(self):
    params = params_dict.ParamsDict({
        'batch_size': 1,
        'model_dir': self._model_dir,
        'train': {
            'optimizer': {
                'type': 'momentum',
                'momentum': 0.9,
            },
            'learning_rate': {
                'type': 'step',
                'init_learning_rate': 0.2,
                'warmup_learning_rate': 0.1,
                'warmup_steps': 100,
                'learning_rate_levels': [0.02, 0.002],
                'learning_rate_steps': [200, 400],
            },
            'checkpoint': {
                'path': '',
                'prefix': '',
                'skip_checkpoint_variables': True,
            },
            'iterations_per_loop': 1,
            'frozen_variable_prefix': 'resnet50_conv2',
        },
        'enable_summary': False,
        'architecture': {
            'use_bfloat16': False,
        },
    })

    def _input_fn(params):
      features = tf.data.Dataset.from_tensor_slices([[1], [2], [3]])
      labels = tf.data.Dataset.from_tensor_slices([[1], [2], [3]])
      data = tf.data.Dataset.zip((features, labels)).repeat()
      dataset = data.batch(params['batch_size'], drop_remainder=True)
      return dataset

    model_factory = DummyModel(params)

    # Use local TPU for testing.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

    with tf.device(''):
      with strategy.scope():
        model = model_factory.build_model()
        metrics = [tf.keras.metrics.MeanSquaredError()]
        loss = model_factory.build_loss_fn()
        model.compile(optimizer=model.optimizer, loss=loss, metrics=metrics)
        model.summary()

      training_steps_per_epoch = 3
      tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=self._model_dir)
      weights_file_path = os.path.join(self._model_dir,
                                       'weights.{epoch:02d}-{val_loss:.2f}.tf')
      checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(weights_file_path)

      training_callbacks = [checkpoint_cb, tensorboard_cb]

      model.fit(
          _input_fn({'batch_size': params.batch_size}),
          epochs=2,
          steps_per_epoch=training_steps_per_epoch,
          callbacks=training_callbacks,
          validation_data=_input_fn({'batch_size': params.batch_size}),
          validation_steps=1,
          validation_freq=1)
      model.evaluate(_input_fn({'batch_size': params.batch_size}), steps=3)

    out_files = tf.io.gfile.glob(os.path.join(self._model_dir, '*'))
    logging.info('Model output files: %s', out_files)
    self.assertNotEmpty(out_files)


if __name__ == '__main__':
  assert tf.version.VERSION.startswith('2.')
  logging.set_verbosity(logging.INFO)
  tf.test.main()

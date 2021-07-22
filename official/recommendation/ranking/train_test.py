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

"""Unit tests for ranking model and associated functionality."""

import json
import os
from absl import flags
from absl.testing import parameterized
import tensorflow as tf

from official.recommendation.ranking import common
from official.recommendation.ranking import train

FLAGS = flags.FLAGS


def _get_params_override(vocab_sizes,
                         interaction='dot',
                         use_orbit=True,
                         strategy='mirrored'):
  # Update `data_dir` if `synthetic_data=False`.
  data_dir = ''

  return json.dumps({
      'runtime': {
          'distribution_strategy': strategy,
      },
      'task': {
          'model': {
              'vocab_sizes': vocab_sizes,
              'embedding_dim': [8] * len(vocab_sizes),
              'bottom_mlp': [64, 32, 8],
              'interaction': interaction,
          },
          'train_data': {
              'input_path': os.path.join(data_dir, 'train/*'),
              'global_batch_size': 16,
          },
          'validation_data': {
              'input_path': os.path.join(data_dir, 'eval/*'),
              'global_batch_size': 16,
          },
          'use_synthetic_data': True,
      },
      'trainer': {
          'use_orbit': use_orbit,
          'validation_interval': 20,
          'validation_steps': 20,
          'train_steps': 40,
      },
  })


class TrainTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._temp_dir = self.get_temp_dir()
    self._model_dir = os.path.join(self._temp_dir, 'model_dir')
    tf.io.gfile.makedirs(self._model_dir)
    FLAGS.model_dir = self._model_dir

    FLAGS.tpu = ''

  def tearDown(self):
    tf.io.gfile.rmtree(self._model_dir)
    super().tearDown()

  @parameterized.named_parameters(
      ('DlrmOneDeviceCTL', 'one_device', 'dot', True),
      ('DlrmOneDevice', 'one_device', 'dot', False),
      ('DcnOneDeviceCTL', 'one_device', 'cross', True),
      ('DcnOneDevice', 'one_device', 'cross', False),
      ('DlrmTPUCTL', 'tpu', 'dot', True),
      ('DlrmTPU', 'tpu', 'dot', False),
      ('DcnTPUCTL', 'tpu', 'cross', True),
      ('DcnTPU', 'tpu', 'cross', False),
      ('DlrmMirroredCTL', 'Mirrored', 'dot', True),
      ('DlrmMirrored', 'Mirrored', 'dot', False),
      ('DcnMirroredCTL', 'Mirrored', 'cross', True),
      ('DcnMirrored', 'Mirrored', 'cross', False),
  )
  def testTrainEval(self, strategy, interaction, use_orbit=True):
    # Set up simple trainer with synthetic data.
    # By default the mode must be `train_and_eval`.
    self.assertEqual(FLAGS.mode, 'train_and_eval')

    vocab_sizes = [40, 12, 11, 13]

    FLAGS.params_override = _get_params_override(vocab_sizes=vocab_sizes,
                                                 interaction=interaction,
                                                 use_orbit=use_orbit,
                                                 strategy=strategy)
    train.main('unused_args')
    self.assertNotEmpty(
        tf.io.gfile.glob(os.path.join(self._model_dir, 'params.yaml')))

  @parameterized.named_parameters(
      ('DlrmTPUCTL', 'tpu', 'dot', True),
      ('DlrmTPU', 'tpu', 'dot', False),
      ('DcnTPUCTL', 'tpu', 'cross', True),
      ('DcnTPU', 'tpu', 'cross', False),
      ('DlrmMirroredCTL', 'Mirrored', 'dot', True),
      ('DlrmMirrored', 'Mirrored', 'dot', False),
      ('DcnMirroredCTL', 'Mirrored', 'cross', True),
      ('DcnMirrored', 'Mirrored', 'cross', False),
  )
  def testTrainThenEval(self, strategy, interaction, use_orbit=True):
    # Set up simple trainer with synthetic data.
    vocab_sizes = [40, 12, 11, 13]

    FLAGS.params_override = _get_params_override(vocab_sizes=vocab_sizes,
                                                 interaction=interaction,
                                                 use_orbit=use_orbit,
                                                 strategy=strategy)
    # Training.
    FLAGS.mode = 'train'
    train.main('unused_args')
    self.assertNotEmpty(
        tf.io.gfile.glob(os.path.join(self._model_dir, 'params.yaml')))

    # Evaluation.
    FLAGS.mode = 'eval'
    train.main('unused_args')


if __name__ == '__main__':
  common.define_flags()
  tf.test.main()

# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
import tensorflow as tf, tf_keras

from official.recommendation.ranking import common
from official.recommendation.ranking import train

FLAGS = flags.FLAGS


def _get_params_override(vocab_sizes,
                         multi_hot_sizes=None,
                         use_multi_hot=False,
                         interaction='dot',
                         use_orbit=True,
                         strategy='mirrored',
                         concat_dense=True,
                         dcn_num_layers=3,
                         dcn_low_rank_dim=64,
                         use_partial_tpu_embedding=True,
                         ):
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
              'concat_dense': concat_dense,
              'dcn_num_layers': dcn_num_layers,
              'dcn_low_rank_dim': dcn_low_rank_dim,
              'use_multi_hot': use_multi_hot,
              'use_partial_tpu_embedding': use_partial_tpu_embedding,
              'multi_hot_sizes': multi_hot_sizes,
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
      ('DlrmOneDeviceCTL', 'one_device', 'dot', True, True, 3, 64, False, True),
      ('DlrmOneDevice', 'one_device', 'dot', False, True, 3, 64, False, True),
      (
          'DcnOneDeviceCTL',
          'one_device',
          'cross',
          True,
          True,
          3,
          64,
          False,
          True,
      ),
      ('DcnOneDevice', 'one_device', 'cross', False, True, 3, 64, False, True),
      (
          'DlrmDcnV2OneDeviceCTL',
          'one_device',
          'multi_layer_dcn',
          True,
          False,
          3,
          64,
          True,
          False,
      ),
      (
          'DlrmDcnV2OneDevice',
          'one_device',
          'multi_layer_dcn',
          False,
          False,
          3,
          64,
          True,
          False,
      ),
      ('DlrmTPUCTL', 'tpu', 'dot', True, True, 3, 64, False, True),
      ('DlrmTPU', 'tpu', 'dot', False, True, 3, 64, False, True),
      ('DcnTPUCTL', 'tpu', 'cross', True, True, 3, 64, False, True),
      ('DcnTPU', 'tpu', 'cross', False, True, 3, 64, False, True),
      (
          'DlrmDcnV2TPUCTL',
          'tpu',
          'multi_layer_dcn',
          True,
          False,
          3,
          64,
          True,
          False,
      ),
      (
          'DlrmDcnV2TPU',
          'tpu',
          'multi_layer_dcn',
          False,
          False,
          3,
          64,
          True,
          False,
      ),
      ('DlrmMirroredCTL', 'Mirrored', 'dot', True, True, 3, 64, False, True),
      ('DlrmMirrored', 'Mirrored', 'dot', False, True, 3, 64, False, True),
      ('DcnMirroredCTL', 'Mirrored', 'cross', True, True, 3, 64, False, True),
      ('DcnMirrored', 'Mirrored', 'cross', False, True, 3, 64, False, True),
      (
          'DlrmDcnV2MirroredCTL',
          'Mirrored',
          'multi_layer_dcn',
          True,
          False,
          3,
          64,
          True,
          False,
      ),
      (
          'DlrmDcnV2Mirrored',
          'Mirrored',
          'multi_layer_dcn',
          False,
          False,
          3,
          64,
          True,
          False,
      ),
  )
  def testTrainEval(
      self,
      strategy,
      interaction,
      use_orbit=True,
      concat_dense=True,
      dcn_num_layers=3,
      dcn_low_rank_dim=64,
      use_multi_hot=False,
      use_partial_tpu_embedding=True,
  ):
    # Set up simple trainer with synthetic data.
    # By default the mode must be `train_and_eval`.
    self.assertEqual(FLAGS.mode, 'train_and_eval')

    vocab_sizes = [40, 12, 11, 13]
    multi_hot_sizes = [1, 2, 3, 1]

    FLAGS.params_override = _get_params_override(
        vocab_sizes=vocab_sizes,
        multi_hot_sizes=multi_hot_sizes,
        use_multi_hot=use_multi_hot,
        interaction=interaction,
        use_orbit=use_orbit,
        strategy=strategy,
        concat_dense=concat_dense,
        dcn_num_layers=dcn_num_layers,
        dcn_low_rank_dim=dcn_low_rank_dim,
        use_partial_tpu_embedding=use_partial_tpu_embedding,
    )

    train.main('unused_args')
    self.assertNotEmpty(
        tf.io.gfile.glob(os.path.join(self._model_dir, 'params.yaml'))
    )

  @parameterized.named_parameters(
      ('DlrmTPUCTL', 'tpu', 'dot', True, True, 3, 64, False, True),
      ('DlrmTPU', 'tpu', 'dot', False, True, 3, 64, False, True),
      ('DcnTPUCTL', 'tpu', 'cross', True, True, 3, 64, False, True),
      ('DcnTPU', 'tpu', 'cross', False, True, 3, 64, False, True),
      (
          'DlrmDcnV2TPUCTL',
          'tpu',
          'multi_layer_dcn',
          True,
          False,
          3,
          64,
          True,
          False,
      ),
      (
          'DlrmDcnV2TPU',
          'tpu',
          'multi_layer_dcn',
          False,
          False,
          3,
          64,
          True,
          False,
      ),
      ('DlrmMirroredCTL', 'Mirrored', 'dot', True, True, 3, 64, False, True),
      ('DlrmMirrored', 'Mirrored', 'dot', False, True, 3, 64, False, True),
      ('DcnMirroredCTL', 'Mirrored', 'cross', True, True, 3, 64, False, True),
      ('DcnMirrored', 'Mirrored', 'cross', False, True, 3, 64, False, True),
      (
          'DlrmDcnV2MirroredCTL',
          'Mirrored',
          'multi_layer_dcn',
          True,
          False,
          3,
          64,
          True,
          False,
      ),
      (
          'DlrmDcnV2Mirrored',
          'Mirrored',
          'multi_layer_dcn',
          False,
          False,
          3,
          64,
          True,
          False,
      ),
  )
  def testTrainThenEval(
      self,
      strategy,
      interaction,
      use_orbit=True,
      concat_dense=True,
      dcn_num_layers=3,
      dcn_low_rank_dim=64,
      use_multi_hot=False,
      use_partial_tpu_embedding=True,
  ):
    # Set up simple trainer with synthetic data.
    vocab_sizes = [40, 12, 11, 13]
    multi_hot_sizes = [1, 2, 3, 1]

    FLAGS.params_override = _get_params_override(
        vocab_sizes=vocab_sizes,
        multi_hot_sizes=multi_hot_sizes,
        interaction=interaction,
        use_orbit=use_orbit,
        strategy=strategy,
        concat_dense=concat_dense,
        dcn_num_layers=dcn_num_layers,
        dcn_low_rank_dim=dcn_low_rank_dim,
        use_multi_hot=use_multi_hot,
        use_partial_tpu_embedding=use_partial_tpu_embedding,
    )

    default_mode = FLAGS.mode
    # Training.
    FLAGS.mode = 'train'
    train.main('unused_args')
    self.assertNotEmpty(
        tf.io.gfile.glob(os.path.join(self._model_dir, 'params.yaml'))
    )

    # Evaluation.
    FLAGS.mode = 'eval'
    train.main('unused_args')
    FLAGS.mode = default_mode


if __name__ == '__main__':
  common.define_flags()
  tf.test.main()

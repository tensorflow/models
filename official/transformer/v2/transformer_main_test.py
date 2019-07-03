# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Test Transformer model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import unittest

from absl import flags
from absl.testing import flagsaver
import tensorflow as tf

from official.transformer.v2 import misc
from official.transformer.v2 import transformer_main as tm
from official.utils.misc import keras_utils

from tensorflow.python.eager import context # pylint: disable=ungrouped-imports

FLAGS = flags.FLAGS
FIXED_TIMESTAMP = 'my_time_stamp'
WEIGHT_PATTERN = re.compile(r'weights-epoch-.+\.hdf5')


def _generate_file(filepath, lines):
  with open(filepath, 'w') as f:
    for l in lines:
      f.write('{}\n'.format(l))


class TransformerTaskTest(tf.test.TestCase):
  local_flags = None

  def setUp(self):
    temp_dir = self.get_temp_dir()
    if TransformerTaskTest.local_flags is None:
      misc.define_transformer_flags()
      # Loads flags, array cannot be blank.
      flags.FLAGS(['foo'])
      TransformerTaskTest.local_flags = flagsaver.save_flag_values()
    else:
      flagsaver.restore_flag_values(TransformerTaskTest.local_flags)
    FLAGS.model_dir = os.path.join(temp_dir, FIXED_TIMESTAMP)
    FLAGS.param_set = 'tiny'
    FLAGS.use_synthetic_data = True
    FLAGS.steps_between_evals = 1
    FLAGS.train_steps = 2
    FLAGS.validation_steps = 1
    FLAGS.batch_size = 8
    FLAGS.num_gpus = 1
    FLAGS.distribution_strategy = 'off'
    FLAGS.dtype = 'fp32'
    self.model_dir = FLAGS.model_dir
    self.temp_dir = temp_dir
    self.vocab_file = os.path.join(temp_dir, 'vocab')
    self.vocab_size = misc.get_model_params(FLAGS.param_set, 0)['vocab_size']
    self.bleu_source = os.path.join(temp_dir, 'bleu_source')
    self.bleu_ref = os.path.join(temp_dir, 'bleu_ref')
    self.orig_policy = tf.keras.mixed_precision.experimental.global_policy()

  def tearDown(self):
    tf.keras.mixed_precision.experimental.set_policy(self.orig_policy)

  def _assert_exists(self, filepath):
    self.assertTrue(os.path.exists(filepath))

  def test_train(self):
    t = tm.TransformerTask(FLAGS)
    t.train()

  @unittest.skipUnless(keras_utils.is_v2_0(), 'TF 2.0 only test.')
  def test_train_static_batch(self):
    FLAGS.static_batch = True
    t = tm.TransformerTask(FLAGS)
    t.train()

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  @unittest.skipUnless(keras_utils.is_v2_0(), 'TF 2.0 only test.')
  def test_train_1_gpu_with_dist_strat(self):
    FLAGS.distribution_strategy = 'one_device'
    t = tm.TransformerTask(FLAGS)
    t.train()

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  @unittest.skipUnless(keras_utils.is_v2_0(), 'TF 2.0 only test.')
  def test_train_2_gpu(self):
    if context.num_gpus() < 2:
      self.skipTest(
          '{} GPUs are not available for this test. {} GPUs are available'.
          format(2, context.num_gpus()))
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.num_gpus = 2
    FLAGS.param_set = 'base'
    t = tm.TransformerTask(FLAGS)
    t.train()

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  @unittest.skipUnless(keras_utils.is_v2_0(), 'TF 2.0 only test.')
  def test_train_2_gpu_fp16(self):
    if context.num_gpus() < 2:
      self.skipTest(
          '{} GPUs are not available for this test. {} GPUs are available'.
          format(2, context.num_gpus()))
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.num_gpus = 2
    FLAGS.param_set = 'base'
    FLAGS.dtype = 'fp16'
    t = tm.TransformerTask(FLAGS)
    t.train()

  def _prepare_files_and_flags(self, *extra_flags):
    # Make log dir.
    if not os.path.exists(self.temp_dir):
      os.makedirs(self.temp_dir)

    # Fake vocab, bleu_source and bleu_ref.
    tokens = [
        "'<pad>'", "'<EOS>'", "'_'", "'a'", "'b'", "'c'", "'d'", "'a_'", "'b_'",
        "'c_'", "'d_'"
    ]
    tokens += ["'{}'".format(i) for i in range(self.vocab_size - len(tokens))]
    _generate_file(self.vocab_file, tokens)
    _generate_file(self.bleu_source, ['a b', 'c d'])
    _generate_file(self.bleu_ref, ['a b', 'd c'])

    # Update flags.
    update_flags = [
        'ignored_program_name',
        '--vocab_file={}'.format(self.vocab_file),
        '--bleu_source={}'.format(self.bleu_source),
        '--bleu_ref={}'.format(self.bleu_ref),
    ]
    if extra_flags:
      update_flags.extend(extra_flags)
    FLAGS(update_flags)

  def test_predict(self):
    self._prepare_files_and_flags()
    t = tm.TransformerTask(FLAGS)
    t.predict()

  def test_predict_fp16(self):
    self._prepare_files_and_flags('--dtype=fp16')
    t = tm.TransformerTask(FLAGS)
    t.predict()

  def test_eval(self):
    self._prepare_files_and_flags()
    t = tm.TransformerTask(FLAGS)
    t.eval()


if __name__ == '__main__':
  tf.test.main()

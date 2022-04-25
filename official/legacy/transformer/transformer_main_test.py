# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Test Transformer model."""

import os
import re
import sys
import unittest

from absl import flags
from absl.testing import flagsaver
import tensorflow as tf
from tensorflow.python.eager import context  # pylint: disable=ungrouped-imports
from official.legacy.transformer import misc
from official.legacy.transformer import transformer_main

FLAGS = flags.FLAGS
FIXED_TIMESTAMP = 'my_time_stamp'
WEIGHT_PATTERN = re.compile(r'weights-epoch-.+\.hdf5')


def _generate_file(filepath, lines):
  with open(filepath, 'w') as f:
    for l in lines:
      f.write('{}\n'.format(l))


class TransformerTaskTest(tf.test.TestCase):
  local_flags = None

  def setUp(self):  # pylint: disable=g-missing-super-call
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
    FLAGS.train_steps = 1
    FLAGS.validation_steps = 1
    FLAGS.batch_size = 4
    FLAGS.max_length = 1
    FLAGS.num_gpus = 1
    FLAGS.distribution_strategy = 'off'
    FLAGS.dtype = 'fp32'
    self.model_dir = FLAGS.model_dir
    self.temp_dir = temp_dir
    self.vocab_file = os.path.join(temp_dir, 'vocab')
    self.vocab_size = misc.get_model_params(FLAGS.param_set, 0)['vocab_size']
    self.bleu_source = os.path.join(temp_dir, 'bleu_source')
    self.bleu_ref = os.path.join(temp_dir, 'bleu_ref')
    self.orig_policy = (
        tf.compat.v2.keras.mixed_precision.global_policy())

  def tearDown(self):  # pylint: disable=g-missing-super-call
    tf.compat.v2.keras.mixed_precision.set_global_policy(self.orig_policy)

  def _assert_exists(self, filepath):
    self.assertTrue(os.path.exists(filepath))

  def test_train_no_dist_strat(self):
    if context.num_gpus() >= 2:
      self.skipTest('No need to test 2+ GPUs without a distribution strategy.')
    t = transformer_main.TransformerTask(FLAGS)
    t.train()

  def test_train_save_full_model(self):
    if context.num_gpus() >= 2:
      self.skipTest('No need to test 2+ GPUs without a distribution strategy.')
    FLAGS.save_weights_only = False
    t = transformer_main.TransformerTask(FLAGS)
    t.train()

  def test_train_static_batch(self):
    if context.num_gpus() >= 2:
      self.skipTest('No need to test 2+ GPUs without a distribution strategy.')
    FLAGS.distribution_strategy = 'one_device'
    if tf.test.is_built_with_cuda():
      FLAGS.num_gpus = 1
    else:
      FLAGS.num_gpus = 0
    FLAGS.static_batch = True
    t = transformer_main.TransformerTask(FLAGS)
    t.train()

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  def test_train_1_gpu_with_dist_strat(self):
    FLAGS.distribution_strategy = 'one_device'
    t = transformer_main.TransformerTask(FLAGS)
    t.train()

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  def test_train_fp16(self):
    FLAGS.distribution_strategy = 'one_device'
    FLAGS.dtype = 'fp16'
    t = transformer_main.TransformerTask(FLAGS)
    t.train()

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  def test_train_2_gpu(self):
    if context.num_gpus() < 2:
      self.skipTest(
          '{} GPUs are not available for this test. {} GPUs are available'
          .format(2, context.num_gpus()))
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.num_gpus = 2
    FLAGS.param_set = 'base'
    t = transformer_main.TransformerTask(FLAGS)
    t.train()

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  def test_train_2_gpu_fp16(self):
    if context.num_gpus() < 2:
      self.skipTest(
          '{} GPUs are not available for this test. {} GPUs are available'
          .format(2, context.num_gpus()))
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.num_gpus = 2
    FLAGS.param_set = 'base'
    FLAGS.dtype = 'fp16'
    t = transformer_main.TransformerTask(FLAGS)
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
    if context.num_gpus() >= 2:
      self.skipTest('No need to test 2+ GPUs without a distribution strategy.')
    self._prepare_files_and_flags()
    t = transformer_main.TransformerTask(FLAGS)
    t.predict()

  @unittest.skipUnless(tf.test.is_built_with_cuda(), 'requires GPU')
  def test_predict_fp16(self):
    if context.num_gpus() >= 2:
      self.skipTest('No need to test 2+ GPUs without a distribution strategy.')
    self._prepare_files_and_flags('--dtype=fp16')
    t = transformer_main.TransformerTask(FLAGS)
    t.predict()

  def test_eval(self):
    if context.num_gpus() >= 2:
      self.skipTest('No need to test 2+ GPUs without a distribution strategy.')
    if 'test_xla' in sys.argv[0]:
      self.skipTest('TODO(xla): Make this test faster under XLA.')
    self._prepare_files_and_flags()
    t = transformer_main.TransformerTask(FLAGS)
    t.eval()


if __name__ == '__main__':
  tf.test.main()

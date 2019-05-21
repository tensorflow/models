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

from absl import flags
import tensorflow as tf

from official.transformer.v2 import misc
from official.transformer.v2 import transformer_main as tm

FLAGS = flags.FLAGS
FIXED_TIMESTAMP = "my_time_stamp"
WEIGHT_PATTERN = re.compile(r"weights-epoch-.+\.hdf5")


def _generate_file(filepath, lines):
  with open(filepath, "w") as f:
    for l in lines:
      f.write("{}\n".format(l))


class TransformerTaskTest(tf.test.TestCase):

  def setUp(self):
    temp_dir = self.get_temp_dir()
    FLAGS.model_dir = temp_dir
    FLAGS.init_logdir_timestamp = FIXED_TIMESTAMP
    FLAGS.param_set = param_set = "tiny"
    FLAGS.use_synthetic_data = True
    FLAGS.steps_per_epoch = 1
    FLAGS.validation_steps = 1
    FLAGS.train_epochs = 1
    FLAGS.batch_size = 8
    FLAGS.init_weight_path = None
    self.cur_log_dir = os.path.join(temp_dir, FIXED_TIMESTAMP)
    self.vocab_file = os.path.join(self.cur_log_dir, "vocab")
    self.vocab_size = misc.get_model_params(param_set, 0)["vocab_size"]
    self.bleu_source = os.path.join(self.cur_log_dir, "bleu_source")
    self.bleu_ref = os.path.join(self.cur_log_dir, "bleu_ref")
    self.flags_file = os.path.join(self.cur_log_dir, "flags")

  def _assert_exists(self, filepath):
    self.assertTrue(os.path.exists(filepath))

  def test_train(self):
    t = tm.TransformerTask(FLAGS)
    t.train()
    # Test model dir.
    self._assert_exists(self.cur_log_dir)
    # Test saving models.
    self._assert_exists(
        os.path.join(self.cur_log_dir, "saves-model-weights.hdf5"))
    self._assert_exists(os.path.join(self.cur_log_dir, "saves-model.hdf5"))

    # Test callbacks:
    # TensorBoard file.
    self._assert_exists(os.path.join(self.cur_log_dir, "logs"))
    # CSVLogger file.
    self._assert_exists(os.path.join(self.cur_log_dir, "result.csv"))
    # Checkpoint file.
    filenames = os.listdir(self.cur_log_dir)
    matched_weight_file = any([WEIGHT_PATTERN.match(f) for f in filenames])
    self.assertTrue(matched_weight_file)

  def _prepare_files_and_flags(self, *extra_flags):
    # Make log dir.
    if not os.path.exists(self.cur_log_dir):
      os.makedirs(self.cur_log_dir)

    # Fake vocab, bleu_source and bleu_ref.
    tokens = [
        "'<pad>'", "'<EOS>'", "'_'", "'a'", "'b'", "'c'", "'d'", "'a_'", "'b_'",
        "'c_'", "'d_'"
    ]
    tokens += ["'{}'".format(i) for i in range(self.vocab_size - len(tokens))]
    _generate_file(self.vocab_file, tokens)
    _generate_file(self.bleu_source, ["a b", "c d"])
    _generate_file(self.bleu_ref, ["a b", "d c"])

    # Update flags.
    update_flags = [
        "ignored_program_name",
        "--vocab_file={}".format(self.vocab_file),
        "--bleu_source={}".format(self.bleu_source),
        "--bleu_ref={}".format(self.bleu_ref),
    ]
    if extra_flags:
      update_flags.extend(extra_flags)
    FLAGS(update_flags)

  def test_predict(self):
    self._prepare_files_and_flags()
    t = tm.TransformerTask(FLAGS)
    t.predict()

  def test_eval(self):
    self._prepare_files_and_flags()
    t = tm.TransformerTask(FLAGS)
    t.eval()


if __name__ == "__main__":
  misc.define_transformer_flags()
  tf.test.main()

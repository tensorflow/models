# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Tests for run_training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from google.protobuf import text_format

import run_training
from protos import seq2label_pb2
import test_utils

FLAGS = flags.FLAGS


class RunTrainingTest(parameterized.TestCase):

  @parameterized.parameters(2, 4, 7)
  def test_wait_until(self, wait_sec):
    end_time = time.time() + wait_sec
    run_training.wait_until(end_time)
    self.assertEqual(round(time.time() - end_time), 0)

  @parameterized.parameters(
      ({}, {'a': 0.7, 'b': 12.3}, 12.3, None,
       {'a': 0.7, 'b': 12.3, 'is_infeasible': False}),
      ({'a': 0.42}, {'b': 24.5}, 24.5, 32.0,
       {'a': 0.42, 'b': 24.5, 'is_infeasible': False}),
      ({'a': 0.503}, {'a': 0.82, 'b': 7.2}, 7.2, 0.1,
       {'a': 0.82, 'b': 7.2, 'is_infeasible': True}),
      ({}, {'a': 0.7, 'b': 12.3}, float('Inf'), None,
       {'a': 0.7, 'b': 12.3, 'is_infeasible': True})
  )
  def test_update_measures(self, measures, new_measures, loss, max_loss,
                           expected):
    run_training.update_measures(measures, new_measures, loss, max_loss)
    self.assertEqual(measures, expected)

  def test_write_measures(self):
    init_time = time.time()
    measures = {
        'global_step': 311448,
        'train_loss': np.float32(18.36),
        'train_weighted_accuracy': np.float32(0.3295),
        'train_accuracy': 0.8243,
        'is_infeasible': False
    }
    tmp_path = os.path.join(FLAGS.test_tmpdir, 'measures.pbtxt')
    run_training.write_measures(measures, tmp_path, init_time)
    experiment_measures = seq2label_pb2.Seq2LabelExperimentMeasures()
    with tf.gfile.Open(tmp_path) as f:
      text_format.Parse(f.read(), experiment_measures)
    self.assertEqual(experiment_measures.checkpoint_path, tmp_path)
    self.assertFalse(experiment_measures.experiment_infeasible)
    self.assertEqual(experiment_measures.steps, measures['global_step'])
    self.assertGreater(experiment_measures.wall_time, 0)
    self.assertEqual(len(experiment_measures.measures), 3)
    for measure in experiment_measures.measures:
      self.assertAlmostEqual(measure.value, measures[measure.name])

  @parameterized.parameters((test_utils.TEST_TARGETS[:1],),
                            (test_utils.TEST_TARGETS,))
  def test_run_training(self, targets):
    """Tests whether the training loop can be run successfully.

    Generates test input files and runs the main driving code.

    Args:
      targets: the targets to train on.
    """
    # Create test input and metadata files.
    num_examples, read_len = 20, 5
    train_file = test_utils.create_tmp_train_file(num_examples, read_len)
    metadata_path = test_utils.create_tmp_metadata(num_examples, read_len)

    # Check that the training loop runs as expected.
    logdir = os.path.join(FLAGS.test_tmpdir, 'train:{}'.format(len(targets)))
    with flagsaver.flagsaver(
        train_files=train_file,
        metadata_path=metadata_path,
        targets=targets,
        logdir=logdir,
        hparams='train_steps=10,min_read_length=5',
        batch_size=10):
      run_training.main(FLAGS)
      # Check training loop ran by confirming existence of a checkpoint file.
      self.assertIsNotNone(tf.train.latest_checkpoint(FLAGS.logdir))
      # Check training loop ran by confiming existence of a measures file.
      self.assertTrue(
          os.path.exists(os.path.join(FLAGS.logdir, 'measures.pbtxt')))


if __name__ == '__main__':
  absltest.main()

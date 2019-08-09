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
"""Tests for boosted_tree."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

# pylint: disable=g-bad-import-order
from official.boosted_trees import train_higgs
from official.utils.misc import keras_utils
from official.utils.testing import integration

TEST_CSV = os.path.join(os.path.dirname(__file__), "train_higgs_test.csv")

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class BaseTest(tf.test.TestCase):
  """Tests for Wide Deep model."""

  @classmethod
  def setUpClass(cls):  # pylint: disable=invalid-name
    super(BaseTest, cls).setUpClass()
    train_higgs.define_train_higgs_flags()

  def setUp(self):
    # Create temporary CSV file
    self.data_dir = self.get_temp_dir()
    data = pd.read_csv(
        TEST_CSV, dtype=np.float32, names=["c%02d" % i for i in range(29)]
    ).as_matrix()
    self.input_npz = os.path.join(self.data_dir, train_higgs.NPZ_FILE)
    # numpy.savez doesn't take gfile.Gfile, so need to write down and copy.
    tmpfile = tempfile.NamedTemporaryFile()
    np.savez_compressed(tmpfile, data=data)
    tf.io.gfile.copy(tmpfile.name, self.input_npz)

  @unittest.skipIf(keras_utils.is_v2_0(), "TF 1.0 only test.")
  def test_read_higgs_data(self):
    """Tests read_higgs_data() function."""
    # Error when a wrong data_dir is given.
    with self.assertRaisesRegexp(RuntimeError, "Error loading data.*"):
      train_data, eval_data = train_higgs.read_higgs_data(
          self.data_dir + "non-existing-path",
          train_start=0, train_count=15, eval_start=15, eval_count=5)

    # Loading fine with the correct data_dir.
    train_data, eval_data = train_higgs.read_higgs_data(
        self.data_dir,
        train_start=0, train_count=15, eval_start=15, eval_count=5)
    self.assertEqual((15, 29), train_data.shape)
    self.assertEqual((5, 29), eval_data.shape)

  @unittest.skipIf(keras_utils.is_v2_0(), "TF 1.0 only test.")
  def test_make_inputs_from_np_arrays(self):
    """Tests make_inputs_from_np_arrays() function."""
    train_data, _ = train_higgs.read_higgs_data(
        self.data_dir,
        train_start=0, train_count=15, eval_start=15, eval_count=5)
    (input_fn, feature_names,
     feature_columns) = train_higgs.make_inputs_from_np_arrays(
         features_np=train_data[:, 1:], label_np=train_data[:, 0:1])

    # Check feature_names.
    self.assertAllEqual(feature_names,
                        ["feature_%02d" % (i+1) for i in range(28)])

    # Check feature columns.
    self.assertEqual(28, len(feature_columns))
    bucketized_column_type = type(
        tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column("feature_01"),
            boundaries=[0, 1, 2]))  # dummy boundaries.
    for feature_column in feature_columns:
      self.assertIsInstance(feature_column, bucketized_column_type)
      # At least 2 boundaries.
      self.assertGreaterEqual(len(feature_column.boundaries), 2)
    # Tests that the source column names of the bucketized columns match.
    self.assertAllEqual(feature_names,
                        [col.source_column.name for col in feature_columns])

    # Check features.
    features, labels = input_fn().make_one_shot_iterator().get_next()
    with tf.Session() as sess:
      features, labels = sess.run((features, labels))
    self.assertIsInstance(features, dict)
    self.assertAllEqual(feature_names, sorted(features.keys()))
    self.assertAllEqual([[15, 1]] * 28,
                        [features[name].shape for name in feature_names])
    # Validate actual values of some features.
    self.assertAllClose(
        [0.869293, 0.907542, 0.798834, 1.344384, 1.105009, 1.595839,
         0.409391, 0.933895, 1.405143, 1.176565, 0.945974, 0.739356,
         1.384097, 1.383548, 1.343652],
        np.squeeze(features[feature_names[0]], 1))
    self.assertAllClose(
        [-0.653674, -0.213641, 1.540659, -0.676015, 1.020974, 0.643109,
         -1.038338, -2.653732, 0.567342, 0.534315, 0.720819, -0.481741,
         1.409523, -0.307865, 1.474605],
        np.squeeze(features[feature_names[10]], 1))

  @unittest.skipIf(keras_utils.is_v2_0(), "TF 1.0 only test.")
  def test_end_to_end(self):
    """Tests end-to-end running."""
    model_dir = os.path.join(self.get_temp_dir(), "model")
    integration.run_synthetic(
        main=train_higgs.main, tmp_root=self.get_temp_dir(), extra_flags=[
            "--data_dir", self.data_dir,
            "--model_dir", model_dir,
            "--n_trees", "5",
            "--train_start", "0",
            "--train_count", "12",
            "--eval_start", "12",
            "--eval_count", "8",
        ],
        synth=False, max_train=None)
    self.assertTrue(tf.gfile.Exists(os.path.join(model_dir, "checkpoint")))

  @unittest.skipIf(keras_utils.is_v2_0(), "TF 1.0 only test.")
  def test_end_to_end_with_export(self):
    """Tests end-to-end running."""
    model_dir = os.path.join(self.get_temp_dir(), "model")
    export_dir = os.path.join(self.get_temp_dir(), "export")
    integration.run_synthetic(
        main=train_higgs.main, tmp_root=self.get_temp_dir(), extra_flags=[
            "--data_dir", self.data_dir,
            "--model_dir", model_dir,
            "--export_dir", export_dir,
            "--n_trees", "5",
            "--train_start", "0",
            "--train_count", "12",
            "--eval_start", "12",
            "--eval_count", "8",
        ],
        synth=False, max_train=None)
    self.assertTrue(tf.gfile.Exists(os.path.join(model_dir, "checkpoint")))
    self.assertTrue(tf.gfile.Exists(os.path.join(export_dir)))


if __name__ == "__main__":
  tf.test.main()

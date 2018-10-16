# Copyright 2018 The TensorFlow Authors.
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

"""Tests for base.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from absl import flags
import numpy as np
import tensorflow as tf

from astrowavenet.data import base

FLAGS = flags.FLAGS

flags.DEFINE_string("test_srcdir", "", "Test source directory.")

_TEST_TFRECORD_FILE = "astrowavenet/data/test_data/test-dataset.tfrecord"


class TFRecordDataset(base.TFRecordDataset):
  """Concrete subclass of TFRecordDataset for testing."""

  @staticmethod
  def default_config():
    config = super(TFRecordDataset, TFRecordDataset).default_config()
    config.update({
        "shuffle_values_buffer": 0,  # Ensure deterministic output.
        "input_dim": 1,
        "conditioning_dim": 1,
        "include_weights": False,
    })
    return config

  def create_example_parser(self):
    """Returns a function that parses a single tf.Example proto."""

    def _example_parser(serialized_example):
      """Parses a single tf.Example into feature and label Tensors."""
      features = tf.parse_single_example(
          serialized_example,
          features={
              "feature_1": tf.VarLenFeature(tf.float32),
              "feature_2": tf.VarLenFeature(tf.float32),
              "feature_3": tf.VarLenFeature(tf.float32),
              "feature_4": tf.VarLenFeature(tf.float32),
              "weights": tf.VarLenFeature(tf.float32),
          })

      output = {}
      if self.config.input_dim == 1:
        # Shape = [num_samples].
        output["autoregressive_input"] = features["feature_1"].values
      elif self.config.input_dim == 2:
        # Shape = [num_samples, 2].
        output["autoregressive_input"] = tf.stack(
            [features["feature_1"].values, features["feature_2"].values],
            axis=-1)
      else:
        raise ValueError("Unexpected input_dim: {}".format(
            self.config.input_dim))

      if self.config.conditioning_dim == 1:
        # Shape = [num_samples].
        output["conditioning_stack"] = features["feature_3"].values
      elif self.config.conditioning_dim == 2:
        # Shape = [num_samples, 2].
        output["conditioning_stack"] = tf.stack(
            [features["feature_3"].values, features["feature_4"].values],
            axis=-1)
      else:
        raise ValueError("Unexpected conditioning_dim: {}".format(
            self.config.conditioning_dim))

      if self.config.include_weights:
        output["weights"] = features["weights"].values

      return output

    return _example_parser


class TFRecordDatasetTest(tf.test.TestCase):

  def setUp(self):
    super(TFRecordDatasetTest, self).setUp()

    # The test dataset contains 8 tensorflow.Example protocol buffers. The i-th
    # Example contains the following features:
    #   feature_1 = range(10, 10 + i + 1)
    #   feature_2 = range(20, 20 + i + 1)
    #   feature_3 = range(30, 30 + i + 1)
    #   feature_4 = range(40, 40 + i + 1)
    #   weights = [0] * i + [1]
    self._file_pattern = os.path.join(FLAGS.test_srcdir, _TEST_TFRECORD_FILE)

  def testTrainMode(self):
    builder = TFRecordDataset(self._file_pattern, tf.estimator.ModeKeys.TRAIN)
    next_features = builder.build(5).make_one_shot_iterator().get_next()
    self.assertItemsEqual(
        ["autoregressive_input", "conditioning_stack", "weights"],
        next_features.keys())

    # Features have dynamic length but fixed batch size and input dimension.
    next_features["autoregressive_input"].shape.assert_is_compatible_with(
        [5, None, 1])
    next_features["conditioning_stack"].shape.assert_is_compatible_with(
        [5, None, 1])
    next_features["weights"].shape.assert_is_compatible_with([5, 1, None])

    # Dataset repeats indefinitely.
    with self.test_session() as sess:
      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [0], [0], [0], [0]],
          [[10], [11], [0], [0], [0]],
          [[10], [11], [12], [0], [0]],
          [[10], [11], [12], [13], [0]],
          [[10], [11], [12], [13], [14]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [0], [0], [0], [0]],
          [[30], [31], [0], [0], [0]],
          [[30], [31], [32], [0], [0]],
          [[30], [31], [32], [33], [0]],
          [[30], [31], [32], [33], [34]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1], [0], [0], [0], [0]],
          [[1], [1], [0], [0], [0]],
          [[1], [1], [1], [0], [0]],
          [[1], [1], [1], [1], [0]],
          [[1], [1], [1], [1], [1]],
      ], features["weights"])

      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [11], [12], [13], [14], [15], [0], [0]],
          [[10], [11], [12], [13], [14], [15], [16], [0]],
          [[10], [11], [12], [13], [14], [15], [16], [17]],
          [[10], [0], [0], [0], [0], [0], [0], [0]],
          [[10], [11], [0], [0], [0], [0], [0], [0]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [31], [32], [33], [34], [35], [0], [0]],
          [[30], [31], [32], [33], [34], [35], [36], [0]],
          [[30], [31], [32], [33], [34], [35], [36], [37]],
          [[30], [0], [0], [0], [0], [0], [0], [0]],
          [[30], [31], [0], [0], [0], [0], [0], [0]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1], [1], [1], [1], [1], [1], [0], [0]],
          [[1], [1], [1], [1], [1], [1], [1], [0]],
          [[1], [1], [1], [1], [1], [1], [1], [1]],
          [[1], [0], [0], [0], [0], [0], [0], [0]],
          [[1], [1], [0], [0], [0], [0], [0], [0]],
      ], features["weights"])

      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [11], [12], [0], [0], [0], [0]],
          [[10], [11], [12], [13], [0], [0], [0]],
          [[10], [11], [12], [13], [14], [0], [0]],
          [[10], [11], [12], [13], [14], [15], [0]],
          [[10], [11], [12], [13], [14], [15], [16]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [31], [32], [0], [0], [0], [0]],
          [[30], [31], [32], [33], [0], [0], [0]],
          [[30], [31], [32], [33], [34], [0], [0]],
          [[30], [31], [32], [33], [34], [35], [0]],
          [[30], [31], [32], [33], [34], [35], [36]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1], [1], [1], [0], [0], [0], [0]],
          [[1], [1], [1], [1], [0], [0], [0]],
          [[1], [1], [1], [1], [1], [0], [0]],
          [[1], [1], [1], [1], [1], [1], [0]],
          [[1], [1], [1], [1], [1], [1], [1]],
      ], features["weights"])

  def testTrainModeReadWeights(self):
    config_overrides = {"include_weights": True}
    builder = TFRecordDataset(
        self._file_pattern,
        tf.estimator.ModeKeys.TRAIN,
        config_overrides=config_overrides)
    next_features = builder.build(5).make_one_shot_iterator().get_next()
    self.assertItemsEqual(
        ["autoregressive_input", "conditioning_stack", "weights"],
        next_features.keys())

    # Features have dynamic length but fixed batch size and input dimension.
    next_features["autoregressive_input"].shape.assert_is_compatible_with(
        [5, None, 1])
    next_features["conditioning_stack"].shape.assert_is_compatible_with(
        [5, None, 1])
    next_features["weights"].shape.assert_is_compatible_with([5, None, 1])

    # Dataset repeats indefinitely.
    with self.test_session() as sess:
      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [0], [0], [0], [0]],
          [[10], [11], [0], [0], [0]],
          [[10], [11], [12], [0], [0]],
          [[10], [11], [12], [13], [0]],
          [[10], [11], [12], [13], [14]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [0], [0], [0], [0]],
          [[30], [31], [0], [0], [0]],
          [[30], [31], [32], [0], [0]],
          [[30], [31], [32], [33], [0]],
          [[30], [31], [32], [33], [34]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1], [0], [0], [0], [0]],
          [[0], [1], [0], [0], [0]],
          [[0], [0], [1], [0], [0]],
          [[0], [0], [0], [1], [0]],
          [[0], [0], [0], [0], [1]],
      ], features["weights"])

      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [11], [12], [13], [14], [15], [0], [0]],
          [[10], [11], [12], [13], [14], [15], [16], [0]],
          [[10], [11], [12], [13], [14], [15], [16], [17]],
          [[10], [0], [0], [0], [0], [0], [0], [0]],
          [[10], [11], [0], [0], [0], [0], [0], [0]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [31], [32], [33], [34], [35], [0], [0]],
          [[30], [31], [32], [33], [34], [35], [36], [0]],
          [[30], [31], [32], [33], [34], [35], [36], [37]],
          [[30], [0], [0], [0], [0], [0], [0], [0]],
          [[30], [31], [0], [0], [0], [0], [0], [0]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[0], [0], [0], [0], [0], [1], [0], [0]],
          [[0], [0], [0], [0], [0], [0], [1], [0]],
          [[0], [0], [0], [0], [0], [0], [0], [1]],
          [[1], [0], [0], [0], [0], [0], [0], [0]],
          [[0], [1], [0], [0], [0], [0], [0], [0]],
      ], features["weights"])

      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [11], [12], [0], [0], [0], [0]],
          [[10], [11], [12], [13], [0], [0], [0]],
          [[10], [11], [12], [13], [14], [0], [0]],
          [[10], [11], [12], [13], [14], [15], [0]],
          [[10], [11], [12], [13], [14], [15], [16]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [31], [32], [0], [0], [0], [0]],
          [[30], [31], [32], [33], [0], [0], [0]],
          [[30], [31], [32], [33], [34], [0], [0]],
          [[30], [31], [32], [33], [34], [35], [0]],
          [[30], [31], [32], [33], [34], [35], [36]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[0], [0], [1], [0], [0], [0], [0]],
          [[0], [0], [0], [1], [0], [0], [0]],
          [[0], [0], [0], [0], [1], [0], [0]],
          [[0], [0], [0], [0], [0], [1], [0]],
          [[0], [0], [0], [0], [0], [0], [1]],
      ], features["weights"])

  def testTrainMode2DInput(self):
    config_overrides = {"input_dim": 2}
    builder = TFRecordDataset(
        self._file_pattern,
        tf.estimator.ModeKeys.TRAIN,
        config_overrides=config_overrides)
    next_features = builder.build(5).make_one_shot_iterator().get_next()
    self.assertItemsEqual(
        ["autoregressive_input", "conditioning_stack", "weights"],
        next_features.keys())

    # Features have dynamic length but fixed batch size and input dimension.
    next_features["autoregressive_input"].shape.assert_is_compatible_with(
        [5, None, 2])
    next_features["conditioning_stack"].shape.assert_is_compatible_with(
        [5, None, 1])
    next_features["weights"].shape.assert_is_compatible_with([5, 1, None])

    # Dataset repeats indefinitely.
    with self.test_session() as sess:
      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10, 20], [0, 0], [0, 0], [0, 0], [0, 0]],
          [[10, 20], [11, 21], [0, 0], [0, 0], [0, 0]],
          [[10, 20], [11, 21], [12, 22], [0, 0], [0, 0]],
          [[10, 20], [11, 21], [12, 22], [13, 23], [0, 0]],
          [[10, 20], [11, 21], [12, 22], [13, 23], [14, 24]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [0], [0], [0], [0]],
          [[30], [31], [0], [0], [0]],
          [[30], [31], [32], [0], [0]],
          [[30], [31], [32], [33], [0]],
          [[30], [31], [32], [33], [34]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1, 1], [0, 0], [0, 0], [0, 0], [0, 0]],
          [[1, 1], [1, 1], [0, 0], [0, 0], [0, 0]],
          [[1, 1], [1, 1], [1, 1], [0, 0], [0, 0]],
          [[1, 1], [1, 1], [1, 1], [1, 1], [0, 0]],
          [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
      ], features["weights"])

      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10, 20], [11, 21], [12, 22], [13, 23], [14, 24], [15, 25], [0, 0],
           [0, 0]],
          [[10, 20], [11, 21], [12, 22], [13, 23], [14, 24], [15, 25], [16, 26],
           [0, 0]],
          [[10, 20], [11, 21], [12, 22], [13, 23], [14, 24], [15, 25], [16, 26],
           [17, 27]],
          [[10, 20], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
          [[10, 20], [11, 21], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [31], [32], [33], [34], [35], [0], [0]],
          [[30], [31], [32], [33], [34], [35], [36], [0]],
          [[30], [31], [32], [33], [34], [35], [36], [37]],
          [[30], [0], [0], [0], [0], [0], [0], [0]],
          [[30], [31], [0], [0], [0], [0], [0], [0]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [0, 0], [0, 0]],
          [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [0, 0]],
          [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
          [[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
          [[1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
      ], features["weights"])

      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10, 20], [11, 21], [12, 22], [0, 0], [0, 0], [0, 0], [0, 0]],
          [[10, 20], [11, 21], [12, 22], [13, 23], [0, 0], [0, 0], [0, 0]],
          [[10, 20], [11, 21], [12, 22], [13, 23], [14, 24], [0, 0], [0, 0]],
          [[10, 20], [11, 21], [12, 22], [13, 23], [14, 24], [15, 25], [0, 0]],
          [[10, 20], [11, 21], [12, 22], [13, 23], [14, 24], [15, 25], [16, 26]
          ],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [31], [32], [0], [0], [0], [0]],
          [[30], [31], [32], [33], [0], [0], [0]],
          [[30], [31], [32], [33], [34], [0], [0]],
          [[30], [31], [32], [33], [34], [35], [0]],
          [[30], [31], [32], [33], [34], [35], [36]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]],
          [[1, 1], [1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0]],
          [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [0, 0], [0, 0]],
          [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [0, 0]],
          [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
      ], features["weights"])

  def testTrainMode2DConditioning(self):
    config_overrides = {"conditioning_dim": 2}
    builder = TFRecordDataset(
        self._file_pattern,
        tf.estimator.ModeKeys.TRAIN,
        config_overrides=config_overrides)
    next_features = builder.build(5).make_one_shot_iterator().get_next()
    self.assertItemsEqual(
        ["autoregressive_input", "conditioning_stack", "weights"],
        next_features.keys())

    # Features have dynamic length but fixed batch size and input dimension.
    next_features["autoregressive_input"].shape.assert_is_compatible_with(
        [5, None, 1])
    next_features["conditioning_stack"].shape.assert_is_compatible_with(
        [5, None, 2])
    next_features["weights"].shape.assert_is_compatible_with([5, 1, None])

    # Dataset repeats indefinitely.
    with self.test_session() as sess:
      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [0], [0], [0], [0]],
          [[10], [11], [0], [0], [0]],
          [[10], [11], [12], [0], [0]],
          [[10], [11], [12], [13], [0]],
          [[10], [11], [12], [13], [14]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30, 40], [0, 0], [0, 0], [0, 0], [0, 0]],
          [[30, 40], [31, 41], [0, 0], [0, 0], [0, 0]],
          [[30, 40], [31, 41], [32, 42], [0, 0], [0, 0]],
          [[30, 40], [31, 41], [32, 42], [33, 43], [0, 0]],
          [[30, 40], [31, 41], [32, 42], [33, 43], [34, 44]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1], [0], [0], [0], [0]],
          [[1], [1], [0], [0], [0]],
          [[1], [1], [1], [0], [0]],
          [[1], [1], [1], [1], [0]],
          [[1], [1], [1], [1], [1]],
      ], features["weights"])

      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [11], [12], [13], [14], [15], [0], [0]],
          [[10], [11], [12], [13], [14], [15], [16], [0]],
          [[10], [11], [12], [13], [14], [15], [16], [17]],
          [[10], [0], [0], [0], [0], [0], [0], [0]],
          [[10], [11], [0], [0], [0], [0], [0], [0]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30, 40], [31, 41], [32, 42], [33, 43], [34, 44], [35, 45], [0, 0],
           [0, 0]],
          [[30, 40], [31, 41], [32, 42], [33, 43], [34, 44], [35, 45], [36, 46],
           [0, 0]],
          [[30, 40], [31, 41], [32, 42], [33, 43], [34, 44], [35, 45], [36, 46],
           [37, 47]],
          [[30, 40], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
          [[30, 40], [31, 41], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1], [1], [1], [1], [1], [1], [0], [0]],
          [[1], [1], [1], [1], [1], [1], [1], [0]],
          [[1], [1], [1], [1], [1], [1], [1], [1]],
          [[1], [0], [0], [0], [0], [0], [0], [0]],
          [[1], [1], [0], [0], [0], [0], [0], [0]],
      ], features["weights"])

      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [11], [12], [0], [0], [0], [0]],
          [[10], [11], [12], [13], [0], [0], [0]],
          [[10], [11], [12], [13], [14], [0], [0]],
          [[10], [11], [12], [13], [14], [15], [0]],
          [[10], [11], [12], [13], [14], [15], [16]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30, 40], [31, 41], [32, 42], [0, 0], [0, 0], [0, 0], [0, 0]],
          [[30, 40], [31, 41], [32, 42], [33, 43], [0, 0], [0, 0], [0, 0]],
          [[30, 40], [31, 41], [32, 42], [33, 43], [34, 44], [0, 0], [0, 0]],
          [[30, 40], [31, 41], [32, 42], [33, 43], [34, 44], [35, 45], [0, 0]],
          [[30, 40], [31, 41], [32, 42], [33, 43], [34, 44], [35, 45], [36, 46]
          ],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1], [1], [1], [0], [0], [0], [0]],
          [[1], [1], [1], [1], [0], [0], [0]],
          [[1], [1], [1], [1], [1], [0], [0]],
          [[1], [1], [1], [1], [1], [1], [0]],
          [[1], [1], [1], [1], [1], [1], [1]],
      ], features["weights"])

  def testTrainModeMaxLength(self):
    config_overrides = {"max_length": 6}
    builder = TFRecordDataset(
        self._file_pattern,
        tf.estimator.ModeKeys.TRAIN,
        config_overrides=config_overrides)
    next_features = builder.build(5).make_one_shot_iterator().get_next()
    self.assertItemsEqual(
        ["autoregressive_input", "conditioning_stack", "weights"],
        next_features.keys())

    # Features have dynamic length but fixed batch size and input dimension.
    next_features["autoregressive_input"].shape.assert_is_compatible_with(
        [5, None, 1])
    next_features["conditioning_stack"].shape.assert_is_compatible_with(
        [5, None, 1])
    next_features["weights"].shape.assert_is_compatible_with([5, 1, None])

    # Dataset repeats indefinitely.
    with self.test_session() as sess:
      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [0], [0], [0], [0]],
          [[10], [11], [0], [0], [0]],
          [[10], [11], [12], [0], [0]],
          [[10], [11], [12], [13], [0]],
          [[10], [11], [12], [13], [14]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [0], [0], [0], [0]],
          [[30], [31], [0], [0], [0]],
          [[30], [31], [32], [0], [0]],
          [[30], [31], [32], [33], [0]],
          [[30], [31], [32], [33], [34]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1], [0], [0], [0], [0]],
          [[1], [1], [0], [0], [0]],
          [[1], [1], [1], [0], [0]],
          [[1], [1], [1], [1], [0]],
          [[1], [1], [1], [1], [1]],
      ], features["weights"])

      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [11], [12], [13], [14], [15]],
          [[10], [11], [12], [13], [14], [15]],
          [[10], [11], [12], [13], [14], [15]],
          [[10], [0], [0], [0], [0], [0]],
          [[10], [11], [0], [0], [0], [0]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [31], [32], [33], [34], [35]],
          [[30], [31], [32], [33], [34], [35]],
          [[30], [31], [32], [33], [34], [35]],
          [[30], [0], [0], [0], [0], [0]],
          [[30], [31], [0], [0], [0], [0]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1], [1], [1], [1], [1], [1]],
          [[1], [1], [1], [1], [1], [1]],
          [[1], [1], [1], [1], [1], [1]],
          [[1], [0], [0], [0], [0], [0]],
          [[1], [1], [0], [0], [0], [0]],
      ], features["weights"])

      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [11], [12], [0], [0], [0]],
          [[10], [11], [12], [13], [0], [0]],
          [[10], [11], [12], [13], [14], [0]],
          [[10], [11], [12], [13], [14], [15]],
          [[10], [11], [12], [13], [14], [15]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [31], [32], [0], [0], [0]],
          [[30], [31], [32], [33], [0], [0]],
          [[30], [31], [32], [33], [34], [0]],
          [[30], [31], [32], [33], [34], [35]],
          [[30], [31], [32], [33], [34], [35]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1], [1], [1], [0], [0], [0]],
          [[1], [1], [1], [1], [0], [0]],
          [[1], [1], [1], [1], [1], [0]],
          [[1], [1], [1], [1], [1], [1]],
          [[1], [1], [1], [1], [1], [1]],
      ], features["weights"])

  def testTrainModeTPU(self):
    config_overrides = {"max_length": 6}
    builder = TFRecordDataset(
        self._file_pattern,
        tf.estimator.ModeKeys.TRAIN,
        config_overrides=config_overrides,
        use_tpu=True)
    next_features = builder.build(5).make_one_shot_iterator().get_next()
    self.assertItemsEqual(
        ["autoregressive_input", "conditioning_stack", "weights"],
        next_features.keys())

    # Features have fixed shape.
    self.assertEqual([5, 6, 1], next_features["autoregressive_input"].shape)
    self.assertEqual([5, 6, 1], next_features["conditioning_stack"].shape)
    self.assertEqual([5, 6, 1], next_features["weights"].shape)

    # Dataset repeats indefinitely.
    with self.test_session() as sess:
      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [0], [0], [0], [0], [0]],
          [[10], [11], [0], [0], [0], [0]],
          [[10], [11], [12], [0], [0], [0]],
          [[10], [11], [12], [13], [0], [0]],
          [[10], [11], [12], [13], [14], [0]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [0], [0], [0], [0], [0]],
          [[30], [31], [0], [0], [0], [0]],
          [[30], [31], [32], [0], [0], [0]],
          [[30], [31], [32], [33], [0], [0]],
          [[30], [31], [32], [33], [34], [0]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1], [0], [0], [0], [0], [0]],
          [[1], [1], [0], [0], [0], [0]],
          [[1], [1], [1], [0], [0], [0]],
          [[1], [1], [1], [1], [0], [0]],
          [[1], [1], [1], [1], [1], [0]],
      ], features["weights"])

      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [11], [12], [13], [14], [15]],
          [[10], [11], [12], [13], [14], [15]],
          [[10], [11], [12], [13], [14], [15]],
          [[10], [0], [0], [0], [0], [0]],
          [[10], [11], [0], [0], [0], [0]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [31], [32], [33], [34], [35]],
          [[30], [31], [32], [33], [34], [35]],
          [[30], [31], [32], [33], [34], [35]],
          [[30], [0], [0], [0], [0], [0]],
          [[30], [31], [0], [0], [0], [0]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1], [1], [1], [1], [1], [1]],
          [[1], [1], [1], [1], [1], [1]],
          [[1], [1], [1], [1], [1], [1]],
          [[1], [0], [0], [0], [0], [0]],
          [[1], [1], [0], [0], [0], [0]],
      ], features["weights"])

      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [11], [12], [0], [0], [0]],
          [[10], [11], [12], [13], [0], [0]],
          [[10], [11], [12], [13], [14], [0]],
          [[10], [11], [12], [13], [14], [15]],
          [[10], [11], [12], [13], [14], [15]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [31], [32], [0], [0], [0]],
          [[30], [31], [32], [33], [0], [0]],
          [[30], [31], [32], [33], [34], [0]],
          [[30], [31], [32], [33], [34], [35]],
          [[30], [31], [32], [33], [34], [35]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1], [1], [1], [0], [0], [0]],
          [[1], [1], [1], [1], [0], [0]],
          [[1], [1], [1], [1], [1], [0]],
          [[1], [1], [1], [1], [1], [1]],
          [[1], [1], [1], [1], [1], [1]],
      ], features["weights"])

  def testEvalMode(self):
    builder = TFRecordDataset(self._file_pattern, tf.estimator.ModeKeys.EVAL)
    next_features = builder.build(5).make_one_shot_iterator().get_next()
    self.assertItemsEqual(
        ["autoregressive_input", "conditioning_stack", "weights"],
        next_features.keys())

    # Features have dynamic length but fixed batch size and input dimension.
    next_features["autoregressive_input"].shape.assert_is_compatible_with(
        [5, None, 1])
    next_features["conditioning_stack"].shape.assert_is_compatible_with(
        [5, None, 1])
    next_features["weights"].shape.assert_is_compatible_with([5, 1, None])

    with self.test_session() as sess:
      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [0], [0], [0], [0]],
          [[10], [11], [0], [0], [0]],
          [[10], [11], [12], [0], [0]],
          [[10], [11], [12], [13], [0]],
          [[10], [11], [12], [13], [14]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [0], [0], [0], [0]],
          [[30], [31], [0], [0], [0]],
          [[30], [31], [32], [0], [0]],
          [[30], [31], [32], [33], [0]],
          [[30], [31], [32], [33], [34]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1], [0], [0], [0], [0]],
          [[1], [1], [0], [0], [0]],
          [[1], [1], [1], [0], [0]],
          [[1], [1], [1], [1], [0]],
          [[1], [1], [1], [1], [1]],
      ], features["weights"])

      # Partial batch.
      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [11], [12], [13], [14], [15], [0], [0]],
          [[10], [11], [12], [13], [14], [15], [16], [0]],
          [[10], [11], [12], [13], [14], [15], [16], [17]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [31], [32], [33], [34], [35], [0], [0]],
          [[30], [31], [32], [33], [34], [35], [36], [0]],
          [[30], [31], [32], [33], [34], [35], [36], [37]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1], [1], [1], [1], [1], [1], [0], [0]],
          [[1], [1], [1], [1], [1], [1], [1], [0]],
          [[1], [1], [1], [1], [1], [1], [1], [1]],
      ], features["weights"])

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(next_features)

  def testEvalModeTPU(self):
    config_overrides = {"max_length": 6}
    builder = TFRecordDataset(
        self._file_pattern,
        tf.estimator.ModeKeys.EVAL,
        config_overrides=config_overrides,
        use_tpu=True)
    next_features = builder.build(5).make_one_shot_iterator().get_next()
    self.assertItemsEqual(
        ["autoregressive_input", "conditioning_stack", "weights"],
        next_features.keys())

    # Features have fixed shape.
    self.assertEqual([5, 6, 1], next_features["autoregressive_input"].shape)
    self.assertEqual([5, 6, 1], next_features["conditioning_stack"].shape)
    self.assertEqual([5, 6, 1], next_features["weights"].shape)

    with self.test_session() as sess:
      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [0], [0], [0], [0], [0]],
          [[10], [11], [0], [0], [0], [0]],
          [[10], [11], [12], [0], [0], [0]],
          [[10], [11], [12], [13], [0], [0]],
          [[10], [11], [12], [13], [14], [0]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [0], [0], [0], [0], [0]],
          [[30], [31], [0], [0], [0], [0]],
          [[30], [31], [32], [0], [0], [0]],
          [[30], [31], [32], [33], [0], [0]],
          [[30], [31], [32], [33], [34], [0]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1], [0], [0], [0], [0], [0]],
          [[1], [1], [0], [0], [0], [0]],
          [[1], [1], [1], [0], [0], [0]],
          [[1], [1], [1], [1], [0], [0]],
          [[1], [1], [1], [1], [1], [0]],
      ], features["weights"])

      # Partial batch, padded.
      features = sess.run(next_features)
      np.testing.assert_almost_equal([
          [[10], [11], [12], [13], [14], [15]],
          [[10], [11], [12], [13], [14], [15]],
          [[10], [11], [12], [13], [14], [15]],
          [[0], [0], [0], [0], [0], [0]],
          [[0], [0], [0], [0], [0], [0]],
      ], features["autoregressive_input"])
      np.testing.assert_almost_equal([
          [[30], [31], [32], [33], [34], [35]],
          [[30], [31], [32], [33], [34], [35]],
          [[30], [31], [32], [33], [34], [35]],
          [[0], [0], [0], [0], [0], [0]],
          [[0], [0], [0], [0], [0], [0]],
      ], features["conditioning_stack"])
      np.testing.assert_almost_equal([
          [[1], [1], [1], [1], [1], [1]],
          [[1], [1], [1], [1], [1], [1]],
          [[1], [1], [1], [1], [1], [1]],
          [[0], [0], [0], [0], [0], [0]],
          [[0], [0], [0], [0], [0], [0]],
      ], features["weights"])

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(next_features)


if __name__ == "__main__":
  tf.test.main()

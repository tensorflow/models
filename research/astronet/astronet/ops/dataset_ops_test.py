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

"""Tests for dataset_ops.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from absl import flags
import numpy as np
import tensorflow as tf

from astronet.ops import dataset_ops
from astronet.util import configdict

FLAGS = flags.FLAGS

flags.DEFINE_string("test_srcdir", "", "Test source directory.")

_TEST_TFRECORD_FILE = "astronet/ops/test_data/test_dataset.tfrecord"


class DatasetOpsTest(tf.test.TestCase):

  def testPadTensorToBatchSize(self):
    with self.test_session():
      # Cannot pad a 0-dimensional Tensor.
      tensor_0d = tf.constant(1)
      with self.assertRaises(ValueError):
        dataset_ops.pad_tensor_to_batch_size(tensor_0d, 10)

      # 1-dimensional Tensor. Un-padded batch size is 5.
      tensor_1d = tf.range(5, dtype=tf.int32)
      self.assertEqual([5], tensor_1d.shape)
      self.assertAllEqual([0, 1, 2, 3, 4], tensor_1d.eval())

      tensor_1d_pad5 = dataset_ops.pad_tensor_to_batch_size(tensor_1d, 5)
      self.assertEqual([5], tensor_1d_pad5.shape)
      self.assertAllEqual([0, 1, 2, 3, 4], tensor_1d_pad5.eval())

      tensor_1d_pad8 = dataset_ops.pad_tensor_to_batch_size(tensor_1d, 8)
      self.assertEqual([8], tensor_1d_pad8.shape)
      self.assertAllEqual([0, 1, 2, 3, 4, 0, 0, 0], tensor_1d_pad8.eval())

      # 2-dimensional Tensor. Un-padded batch size is 3.
      tensor_2d = tf.reshape(tf.range(9, dtype=tf.int32), [3, 3])
      self.assertEqual([3, 3], tensor_2d.shape)
      self.assertAllEqual([[0, 1, 2], [3, 4, 5], [6, 7, 8]], tensor_2d.eval())

      tensor_2d_pad3 = dataset_ops.pad_tensor_to_batch_size(tensor_2d, 3)
      self.assertEqual([3, 3], tensor_2d_pad3.shape)
      self.assertAllEqual([[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                          tensor_2d_pad3.eval())

      tensor_2d_pad4 = dataset_ops.pad_tensor_to_batch_size(tensor_2d, 4)
      self.assertEqual([4, 3], tensor_2d_pad4.shape)
      self.assertAllEqual([[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 0, 0]],
                          tensor_2d_pad4.eval())

  def testPadDatasetToBatchSizeNoWeights(self):
    values = {"labels": np.arange(10, dtype=np.int32)}
    dataset = tf.data.Dataset.from_tensor_slices(values).batch(4)
    self.assertItemsEqual(["labels"], dataset.output_shapes.keys())
    self.assertFalse(dataset.output_shapes["labels"].is_fully_defined())

    dataset_pad = dataset_ops.pad_dataset_to_batch_size(dataset, 4)
    self.assertItemsEqual(["labels", "weights"],
                          dataset_pad.output_shapes.keys())
    self.assertEqual([4], dataset_pad.output_shapes["labels"])
    self.assertEqual([4], dataset_pad.output_shapes["weights"])

    next_batch = dataset_pad.make_one_shot_iterator().get_next()
    next_labels = next_batch["labels"]
    next_weights = next_batch["weights"]

    with self.test_session() as sess:
      labels, weights = sess.run([next_labels, next_weights])
      self.assertAllEqual([0, 1, 2, 3], labels)
      self.assertAllClose([1, 1, 1, 1], weights)

      labels, weights = sess.run([next_labels, next_weights])
      self.assertAllEqual([4, 5, 6, 7], labels)
      self.assertAllClose([1, 1, 1, 1], weights)

      labels, weights = sess.run([next_labels, next_weights])
      self.assertAllEqual([8, 9, 0, 0], labels)
      self.assertAllClose([1, 1, 0, 0], weights)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run([next_labels, next_weights])

  def testPadDatasetToBatchSizeWithWeights(self):
    values = {
        "labels": np.arange(10, dtype=np.int32),
        "weights": 100 + np.arange(10, dtype=np.int32)
    }
    dataset = tf.data.Dataset.from_tensor_slices(values).batch(4)
    self.assertItemsEqual(["labels", "weights"], dataset.output_shapes.keys())
    self.assertFalse(dataset.output_shapes["labels"].is_fully_defined())
    self.assertFalse(dataset.output_shapes["weights"].is_fully_defined())

    dataset_pad = dataset_ops.pad_dataset_to_batch_size(dataset, 4)
    self.assertItemsEqual(["labels", "weights"],
                          dataset_pad.output_shapes.keys())
    self.assertEqual([4], dataset_pad.output_shapes["labels"])
    self.assertEqual([4], dataset_pad.output_shapes["weights"])

    next_batch = dataset_pad.make_one_shot_iterator().get_next()
    next_labels = next_batch["labels"]
    next_weights = next_batch["weights"]

    with self.test_session() as sess:
      labels, weights = sess.run([next_labels, next_weights])
      self.assertAllEqual([0, 1, 2, 3], labels)
      self.assertAllEqual([100, 101, 102, 103], weights)

      labels, weights = sess.run([next_labels, next_weights])
      self.assertAllEqual([4, 5, 6, 7], labels)
      self.assertAllEqual([104, 105, 106, 107], weights)

      labels, weights = sess.run([next_labels, next_weights])
      self.assertAllEqual([8, 9, 0, 0], labels)
      self.assertAllEqual([108, 109, 0, 0], weights)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run([next_labels, next_weights])

  def testSetBatchSizeSingleTensor1d(self):
    dataset = tf.data.Dataset.range(4).batch(2)
    self.assertFalse(dataset.output_shapes.is_fully_defined())

    dataset = dataset_ops.set_batch_size(dataset, 2)
    self.assertEqual([2], dataset.output_shapes)

    next_batch = dataset.make_one_shot_iterator().get_next()
    with self.test_session() as sess:
      batch_value = sess.run(next_batch)
      self.assertAllEqual([0, 1], batch_value)

      batch_value = sess.run(next_batch)
      self.assertAllEqual([2, 3], batch_value)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(next_batch)

  def testSetBatchSizeSingleTensor2d(self):
    values = np.arange(12, dtype=np.int32).reshape([4, 3])
    dataset = tf.data.Dataset.from_tensor_slices(values).batch(2)
    self.assertFalse(dataset.output_shapes.is_fully_defined())

    dataset = dataset_ops.set_batch_size(dataset, 2)
    self.assertEqual([2, 3], dataset.output_shapes)

    next_batch = dataset.make_one_shot_iterator().get_next()
    with self.test_session() as sess:
      batch_value = sess.run(next_batch)
      self.assertAllEqual([[0, 1, 2], [3, 4, 5]], batch_value)

      batch_value = sess.run(next_batch)
      self.assertAllEqual([[6, 7, 8], [9, 10, 11]], batch_value)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(next_batch)

  def testSetBatchSizeNested(self):
    values = {
        "a": 100 + np.arange(4, dtype=np.int32),
        "nest": {
            "b": np.arange(12, dtype=np.int32).reshape([4, 3]),
            "c": np.arange(4, dtype=np.int32)
        }
    }
    dataset = tf.data.Dataset.from_tensor_slices(values).batch(2)
    self.assertItemsEqual(["a", "nest"], dataset.output_shapes.keys())
    self.assertItemsEqual(["b", "c"], dataset.output_shapes["nest"].keys())
    self.assertFalse(dataset.output_shapes["a"].is_fully_defined())
    self.assertFalse(dataset.output_shapes["nest"]["b"].is_fully_defined())
    self.assertFalse(dataset.output_shapes["nest"]["c"].is_fully_defined())

    dataset = dataset_ops.set_batch_size(dataset, 2)
    self.assertItemsEqual(["a", "nest"], dataset.output_shapes.keys())
    self.assertItemsEqual(["b", "c"], dataset.output_shapes["nest"].keys())
    self.assertEqual([2], dataset.output_shapes["a"])
    self.assertEqual([2, 3], dataset.output_shapes["nest"]["b"])
    self.assertEqual([2], dataset.output_shapes["nest"]["c"])

    next_batch = dataset.make_one_shot_iterator().get_next()
    next_a = next_batch["a"]
    next_b = next_batch["nest"]["b"]
    next_c = next_batch["nest"]["c"]

    with self.test_session() as sess:
      a, b, c = sess.run([next_a, next_b, next_c])
      self.assertAllEqual([100, 101], a)
      self.assertAllEqual([[0, 1, 2], [3, 4, 5]], b)
      self.assertAllEqual([0, 1], c)

      a, b, c = sess.run([next_a, next_b, next_c])
      self.assertAllEqual([102, 103], a)
      self.assertAllEqual([[6, 7, 8], [9, 10, 11]], b)
      self.assertAllEqual([2, 3], c)

      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(next_batch)


class BuildDatasetTest(tf.test.TestCase):

  def setUp(self):
    super(BuildDatasetTest, self).setUp()

    # The test dataset contains 10 tensorflow.Example protocol buffers. The i-th
    # Example contains the following features:
    #   global_view = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    #   local_view = [0.0, 1.0, 2.0, 3.0]
    #   aux_feature = 100 + i
    #   label_str = "PC" if i % 3 == 0 else "AFP" if i % 3 == 1 else "NTP"
    self._file_pattern = os.path.join(FLAGS.test_srcdir, _TEST_TFRECORD_FILE)

    self._input_config = configdict.ConfigDict({
        "features": {
            "global_view": {
                "is_time_series": True,
                "length": 8
            },
            "local_view": {
                "is_time_series": True,
                "length": 4
            },
            "aux_feature": {
                "is_time_series": False,
                "length": 1
            }
        }
    })

  def testNonExistentFileRaisesValueError(self):
    with self.assertRaises(ValueError):
      dataset_ops.build_dataset(
          file_pattern="nonexistent",
          input_config=self._input_config,
          batch_size=4)

  def testBuildWithoutLabels(self):
    dataset = dataset_ops.build_dataset(
        file_pattern=self._file_pattern,
        input_config=self._input_config,
        batch_size=4,
        include_labels=False)

    # We can use a one-shot iterator without labels because we don't have the
    # stateful hash map for label ids.
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

    # Expect features only.
    self.assertItemsEqual(["time_series_features", "aux_features"],
                          features.keys())

    with self.test_session() as sess:
      # Batch 1.
      f = sess.run(features)
      np.testing.assert_array_almost_equal([
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
      ], f["time_series_features"]["global_view"])
      np.testing.assert_array_almost_equal([
          [0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3],
      ], f["time_series_features"]["local_view"])
      np.testing.assert_array_almost_equal([[100], [101], [102], [103]],
                                           f["aux_features"]["aux_feature"])

      # Batch 2.
      f = sess.run(features)
      np.testing.assert_array_almost_equal([
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
      ], f["time_series_features"]["global_view"])
      np.testing.assert_array_almost_equal([
          [0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3],
      ], f["time_series_features"]["local_view"])
      np.testing.assert_array_almost_equal([[104], [105], [106], [107]],
                                           f["aux_features"]["aux_feature"])

      # Batch 3.
      f = sess.run(features)
      np.testing.assert_array_almost_equal([
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
      ], f["time_series_features"]["global_view"])
      np.testing.assert_array_almost_equal([
          [0, 1, 2, 3],
          [0, 1, 2, 3],
      ], f["time_series_features"]["local_view"])
      np.testing.assert_array_almost_equal([[108], [109]],
                                           f["aux_features"]["aux_feature"])

      # No more batches.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(features)

  def testLabels1(self):
    self._input_config["label_feature"] = "label_str"
    self._input_config["label_map"] = {"PC": 0, "AFP": 1, "NTP": 2}

    dataset = dataset_ops.build_dataset(
        file_pattern=self._file_pattern,
        input_config=self._input_config,
        batch_size=4)

    # We need an initializable iterator when using labels because of the
    # stateful label id hash table.
    iterator = dataset.make_initializable_iterator()
    inputs = iterator.get_next()
    init_op = tf.tables_initializer()

    # Expect features and labels.
    self.assertItemsEqual(["time_series_features", "aux_features", "labels"],
                          inputs.keys())
    labels = inputs["labels"]

    with self.test_session() as sess:
      sess.run([init_op, iterator.initializer])

      # Fetch 3 batches.
      np.testing.assert_array_equal([0, 1, 2, 0], sess.run(labels))
      np.testing.assert_array_equal([1, 2, 0, 1], sess.run(labels))
      np.testing.assert_array_equal([2, 0], sess.run(labels))

      # No more batches.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(labels)

  def testLabels2(self):
    self._input_config["label_feature"] = "label_str"
    self._input_config["label_map"] = {"PC": 1, "AFP": 0, "NTP": 0}

    dataset = dataset_ops.build_dataset(
        file_pattern=self._file_pattern,
        input_config=self._input_config,
        batch_size=4)

    # We need an initializable iterator when using labels because of the
    # stateful label id hash table.
    iterator = dataset.make_initializable_iterator()
    inputs = iterator.get_next()
    init_op = tf.tables_initializer()

    # Expect features and labels.
    self.assertItemsEqual(["time_series_features", "aux_features", "labels"],
                          inputs.keys())
    labels = inputs["labels"]

    with self.test_session() as sess:
      sess.run([init_op, iterator.initializer])

      # Fetch 3 batches.
      np.testing.assert_array_equal([1, 0, 0, 1], sess.run(labels))
      np.testing.assert_array_equal([0, 0, 1, 0], sess.run(labels))
      np.testing.assert_array_equal([0, 1], sess.run(labels))

      # No more batches.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(labels)

  def testBadLabelIdsRaisesValueError(self):
    self._input_config["label_feature"] = "label_str"

    # Label ids should be contiguous integers starting at 0.
    self._input_config["label_map"] = {"PC": 1, "AFP": 2, "NTP": 3}

    with self.assertRaises(ValueError):
      dataset_ops.build_dataset(
          file_pattern=self._file_pattern,
          input_config=self._input_config,
          batch_size=4)

  def testUnknownLabel(self):
    self._input_config["label_feature"] = "label_str"

    # label_map does not include "NTP".
    self._input_config["label_map"] = {"PC": 1, "AFP": 0}

    dataset = dataset_ops.build_dataset(
        file_pattern=self._file_pattern,
        input_config=self._input_config,
        batch_size=4)

    # We need an initializable iterator when using labels because of the
    # stateful label id hash table.
    iterator = dataset.make_initializable_iterator()
    inputs = iterator.get_next()
    init_op = tf.tables_initializer()

    # Expect features and labels.
    self.assertItemsEqual(["time_series_features", "aux_features", "labels"],
                          inputs.keys())
    labels = inputs["labels"]

    with self.test_session() as sess:
      sess.run([init_op, iterator.initializer])

      # Unknown label "NTP".
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(labels)

  def testReverseTimeSeries(self):
    dataset = dataset_ops.build_dataset(
        file_pattern=self._file_pattern,
        input_config=self._input_config,
        batch_size=4,
        reverse_time_series_prob=1,
        include_labels=False)

    # We can use a one-shot iterator without labels because we don't have the
    # stateful hash map for label ids.
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

    # Expect features only.
    self.assertItemsEqual(["time_series_features", "aux_features"],
                          features.keys())

    with self.test_session() as sess:
      # Batch 1.
      f = sess.run(features)
      np.testing.assert_array_almost_equal([
          [7, 6, 5, 4, 3, 2, 1, 0],
          [7, 6, 5, 4, 3, 2, 1, 0],
          [7, 6, 5, 4, 3, 2, 1, 0],
          [7, 6, 5, 4, 3, 2, 1, 0],
      ], f["time_series_features"]["global_view"])
      np.testing.assert_array_almost_equal([
          [3, 2, 1, 0],
          [3, 2, 1, 0],
          [3, 2, 1, 0],
          [3, 2, 1, 0],
      ], f["time_series_features"]["local_view"])
      np.testing.assert_array_almost_equal([[100], [101], [102], [103]],
                                           f["aux_features"]["aux_feature"])

      # Batch 2.
      f = sess.run(features)
      np.testing.assert_array_almost_equal([
          [7, 6, 5, 4, 3, 2, 1, 0],
          [7, 6, 5, 4, 3, 2, 1, 0],
          [7, 6, 5, 4, 3, 2, 1, 0],
          [7, 6, 5, 4, 3, 2, 1, 0],
      ], f["time_series_features"]["global_view"])
      np.testing.assert_array_almost_equal([
          [3, 2, 1, 0],
          [3, 2, 1, 0],
          [3, 2, 1, 0],
          [3, 2, 1, 0],
      ], f["time_series_features"]["local_view"])
      np.testing.assert_array_almost_equal([[104], [105], [106], [107]],
                                           f["aux_features"]["aux_feature"])

      # Batch 3.
      f = sess.run(features)
      np.testing.assert_array_almost_equal([
          [7, 6, 5, 4, 3, 2, 1, 0],
          [7, 6, 5, 4, 3, 2, 1, 0],
      ], f["time_series_features"]["global_view"])
      np.testing.assert_array_almost_equal([
          [3, 2, 1, 0],
          [3, 2, 1, 0],
      ], f["time_series_features"]["local_view"])
      np.testing.assert_array_almost_equal([[108], [109]],
                                           f["aux_features"]["aux_feature"])

      # No more batches.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(features)

  def testRepeat(self):
    dataset = dataset_ops.build_dataset(
        file_pattern=self._file_pattern,
        input_config=self._input_config,
        batch_size=4,
        include_labels=False)

    # We can use a one-shot iterator without labels because we don't have the
    # stateful hash map for label ids.
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

    # Expect features only.
    self.assertItemsEqual(["time_series_features", "aux_features"],
                          features.keys())

    with self.test_session() as sess:
      # Batch 1.
      f = sess.run(features)
      np.testing.assert_array_almost_equal([
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
      ], f["time_series_features"]["global_view"])
      np.testing.assert_array_almost_equal([
          [0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3],
      ], f["time_series_features"]["local_view"])
      np.testing.assert_array_almost_equal([[100], [101], [102], [103]],
                                           f["aux_features"]["aux_feature"])

      # Batch 2.
      f = sess.run(features)
      np.testing.assert_array_almost_equal([
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
      ], f["time_series_features"]["global_view"])
      np.testing.assert_array_almost_equal([
          [0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3],
      ], f["time_series_features"]["local_view"])
      np.testing.assert_array_almost_equal([[104], [105], [106], [107]],
                                           f["aux_features"]["aux_feature"])

      # Batch 3.
      f = sess.run(features)
      np.testing.assert_array_almost_equal([
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
      ], f["time_series_features"]["global_view"])
      np.testing.assert_array_almost_equal([
          [0, 1, 2, 3],
          [0, 1, 2, 3],
      ], f["time_series_features"]["local_view"])
      np.testing.assert_array_almost_equal([[108], [109]],
                                           f["aux_features"]["aux_feature"])

      # No more batches.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(features)

  def testTPU(self):
    dataset = dataset_ops.build_dataset(
        file_pattern=self._file_pattern,
        input_config=self._input_config,
        batch_size=4,
        include_labels=False)

    # We can use a one-shot iterator without labels because we don't have the
    # stateful hash map for label ids.
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

    # Expect features only.
    self.assertItemsEqual(["time_series_features", "aux_features"],
                          features.keys())

    with self.test_session() as sess:
      # Batch 1.
      f = sess.run(features)
      np.testing.assert_array_almost_equal([
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
      ], f["time_series_features"]["global_view"])
      np.testing.assert_array_almost_equal([
          [0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3],
      ], f["time_series_features"]["local_view"])
      np.testing.assert_array_almost_equal([[100], [101], [102], [103]],
                                           f["aux_features"]["aux_feature"])

      # Batch 2.
      f = sess.run(features)
      np.testing.assert_array_almost_equal([
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
      ], f["time_series_features"]["global_view"])
      np.testing.assert_array_almost_equal([
          [0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3],
          [0, 1, 2, 3],
      ], f["time_series_features"]["local_view"])
      np.testing.assert_array_almost_equal([[104], [105], [106], [107]],
                                           f["aux_features"]["aux_feature"])

      # Batch 3.
      f = sess.run(features)
      np.testing.assert_array_almost_equal([
          [0, 1, 2, 3, 4, 5, 6, 7],
          [0, 1, 2, 3, 4, 5, 6, 7],
      ], f["time_series_features"]["global_view"])
      np.testing.assert_array_almost_equal([
          [0, 1, 2, 3],
          [0, 1, 2, 3],
      ], f["time_series_features"]["local_view"])
      np.testing.assert_array_almost_equal([[108], [109]],
                                           f["aux_features"]["aux_feature"])

      # No more batches.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(features)


if __name__ == "__main__":
  tf.test.main()

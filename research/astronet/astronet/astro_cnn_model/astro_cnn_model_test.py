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

"""Tests for astro_cnn_model.AstroCNNModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from astronet.astro_cnn_model import astro_cnn_model
from astronet.astro_cnn_model import configurations
from astronet.ops import input_ops
from astronet.ops import testing
from astronet.util import configdict


class AstroCNNModelTest(tf.test.TestCase):

  def assertShapeEquals(self, shape, tensor_or_array):
    """Asserts that a Tensor or Numpy array has the expected shape.

    Args:
      shape: Numpy array or anything that can be converted to one.
      tensor_or_array: tf.Tensor, tf.Variable, Numpy array or anything that can
          be converted to one.
    """
    if isinstance(tensor_or_array, (np.ndarray, np.generic)):
      self.assertAllEqual(shape, tensor_or_array.shape)
    elif isinstance(tensor_or_array, (tf.Tensor, tf.Variable)):
      self.assertAllEqual(shape, tensor_or_array.shape.as_list())
    else:
      raise TypeError("tensor_or_array must be a Tensor or Numpy ndarray")

  def testOneTimeSeriesFeature(self):
    # Build config.
    feature_spec = {
        "time_feature_1": {
            "length": 20,
            "is_time_series": True,
        }
    }
    hidden_spec = {
        "time_feature_1": {
            "cnn_num_blocks": 2,
            "cnn_block_size": 2,
            "cnn_initial_num_filters": 4,
            "cnn_block_filter_factor": 1.5,
            "cnn_kernel_size": 3,
            "convolution_padding": "same",
            "pool_size": 2,
            "pool_strides": 2,
        }
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config["hparams"]["time_series_hidden"] = hidden_spec
    config = configdict.ConfigDict(config)

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_cnn_model.AstroCNNModel(features, labels, config.hparams,
                                          tf.estimator.ModeKeys.TRAIN)
    model.build()

    # Validate Tensor shapes.
    block_1_conv_1 = testing.get_variable_by_name(
        "time_feature_1_hidden/block_1/conv_1/kernel")
    self.assertShapeEquals((3, 1, 4), block_1_conv_1)

    block_1_conv_2 = testing.get_variable_by_name(
        "time_feature_1_hidden/block_1/conv_2/kernel")
    self.assertShapeEquals((3, 4, 4), block_1_conv_2)

    block_2_conv_1 = testing.get_variable_by_name(
        "time_feature_1_hidden/block_2/conv_1/kernel")
    self.assertShapeEquals((3, 4, 6), block_2_conv_1)

    block_2_conv_2 = testing.get_variable_by_name(
        "time_feature_1_hidden/block_2/conv_2/kernel")
    self.assertShapeEquals((3, 6, 6), block_2_conv_2)

    self.assertItemsEqual(["time_feature_1"],
                          model.time_series_hidden_layers.keys())
    self.assertShapeEquals((None, 30),
                           model.time_series_hidden_layers["time_feature_1"])
    self.assertEqual(len(model.aux_hidden_layers), 0)
    self.assertIs(model.time_series_hidden_layers["time_feature_1"],
                  model.pre_logits_concat)

    # Execute the TensorFlow graph.
    scaffold = tf.train.Scaffold()
    scaffold.finalize()
    with self.test_session() as sess:
      sess.run([scaffold.init_op, scaffold.local_init_op])
      step = sess.run(model.global_step)
      self.assertEqual(0, step)

      # Fetch predictions.
      features = testing.fake_features(feature_spec, batch_size=16)
      labels = testing.fake_labels(config.hparams.output_dim, batch_size=16)
      feed_dict = input_ops.prepare_feed_dict(model, features, labels)
      predictions = sess.run(model.predictions, feed_dict=feed_dict)
      self.assertShapeEquals((16, 1), predictions)

  def testTwoTimeSeriesFeatures(self):
    # Build config.
    feature_spec = {
        "time_feature_1": {
            "length": 20,
            "is_time_series": True,
        },
        "time_feature_2": {
            "length": 5,
            "is_time_series": True,
        },
        "aux_feature_1": {
            "length": 1,
            "is_time_series": False,
        },
    }
    hidden_spec = {
        "time_feature_1": {
            "cnn_num_blocks": 2,
            "cnn_block_size": 2,
            "cnn_initial_num_filters": 4,
            "cnn_block_filter_factor": 1.5,
            "cnn_kernel_size": 3,
            "convolution_padding": "same",
            "pool_size": 2,
            "pool_strides": 2,
        },
        "time_feature_2": {
            "cnn_num_blocks": 1,
            "cnn_block_size": 1,
            "cnn_initial_num_filters": 5,
            "cnn_block_filter_factor": 1,
            "cnn_kernel_size": 2,
            "convolution_padding": "same",
            "pool_size": 0,
            "pool_strides": 0,
        }
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config["hparams"]["time_series_hidden"] = hidden_spec
    config = configdict.ConfigDict(config)

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_cnn_model.AstroCNNModel(features, labels, config.hparams,
                                          tf.estimator.ModeKeys.TRAIN)
    model.build()

    # Validate Tensor shapes.
    feature_1_block_1_conv_1 = testing.get_variable_by_name(
        "time_feature_1_hidden/block_1/conv_1/kernel")
    self.assertShapeEquals((3, 1, 4), feature_1_block_1_conv_1)

    feature_1_block_1_conv_2 = testing.get_variable_by_name(
        "time_feature_1_hidden/block_1/conv_2/kernel")
    self.assertShapeEquals((3, 4, 4), feature_1_block_1_conv_2)

    feature_1_block_2_conv_1 = testing.get_variable_by_name(
        "time_feature_1_hidden/block_2/conv_1/kernel")
    self.assertShapeEquals((3, 4, 6), feature_1_block_2_conv_1)

    feature_1_block_2_conv_2 = testing.get_variable_by_name(
        "time_feature_1_hidden/block_2/conv_2/kernel")
    self.assertShapeEquals((3, 6, 6), feature_1_block_2_conv_2)

    feature_2_block_1_conv_1 = testing.get_variable_by_name(
        "time_feature_2_hidden/block_1/conv_1/kernel")
    self.assertShapeEquals((2, 1, 5), feature_2_block_1_conv_1)

    self.assertItemsEqual(["time_feature_1", "time_feature_2"],
                          model.time_series_hidden_layers.keys())
    self.assertShapeEquals((None, 30),
                           model.time_series_hidden_layers["time_feature_1"])
    self.assertShapeEquals((None, 25),
                           model.time_series_hidden_layers["time_feature_2"])
    self.assertItemsEqual(["aux_feature_1"], model.aux_hidden_layers.keys())
    self.assertIs(model.aux_features["aux_feature_1"],
                  model.aux_hidden_layers["aux_feature_1"])
    self.assertShapeEquals((None, 56), model.pre_logits_concat)

    # Execute the TensorFlow graph.
    scaffold = tf.train.Scaffold()
    scaffold.finalize()
    with self.test_session() as sess:
      sess.run([scaffold.init_op, scaffold.local_init_op])
      step = sess.run(model.global_step)
      self.assertEqual(0, step)

      # Fetch predictions.
      features = testing.fake_features(feature_spec, batch_size=16)
      labels = testing.fake_labels(config.hparams.output_dim, batch_size=16)
      feed_dict = input_ops.prepare_feed_dict(model, features, labels)
      predictions = sess.run(model.predictions, feed_dict=feed_dict)
      self.assertShapeEquals((16, 1), predictions)


if __name__ == "__main__":
  tf.test.main()

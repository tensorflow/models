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

"""Tests for astro_fc_model.AstroFCModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from astronet.astro_fc_model import astro_fc_model
from astronet.astro_fc_model import configurations
from astronet.ops import input_ops
from astronet.ops import testing
from astronet.util import configdict


class AstroFCModelTest(tf.test.TestCase):

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
            "length": 14,
            "is_time_series": True,
        }
    }
    hidden_spec = {
        "time_feature_1": {
            "num_local_layers": 2,
            "local_layer_size": 20,
            "translation_delta": 2,
            "pooling_type": "max",
            "dropout_rate": 0.5,
        }
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config["hparams"]["time_series_hidden"] = hidden_spec
    config = configdict.ConfigDict(config)

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_fc_model.AstroFCModel(features, labels, config.hparams,
                                        tf.estimator.ModeKeys.TRAIN)
    model.build()

    # Validate Tensor shapes.
    conv = testing.get_variable_by_name("time_feature_1_hidden/conv1d/kernel")
    self.assertShapeEquals((10, 1, 20), conv)

    fc_1 = testing.get_variable_by_name(
        "time_feature_1_hidden/fully_connected_1/weights")
    self.assertShapeEquals((20, 20), fc_1)

    self.assertItemsEqual(["time_feature_1"],
                          model.time_series_hidden_layers.keys())
    self.assertShapeEquals((None, 20),
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
            "num_local_layers": 1,
            "local_layer_size": 20,
            "translation_delta": 1,
            "pooling_type": "max",
            "dropout_rate": 0.5,
        },
        "time_feature_2": {
            "num_local_layers": 2,
            "local_layer_size": 7,
            "translation_delta": 0,
            "dropout_rate": 0,
        }
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config["hparams"]["time_series_hidden"] = hidden_spec
    config = configdict.ConfigDict(config)

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_fc_model.AstroFCModel(features, labels, config.hparams,
                                        tf.estimator.ModeKeys.TRAIN)
    model.build()

    # Validate Tensor shapes.
    conv = testing.get_variable_by_name("time_feature_1_hidden/conv1d/kernel")
    self.assertShapeEquals((18, 1, 20), conv)

    fc_1 = testing.get_variable_by_name(
        "time_feature_2_hidden/fully_connected_1/weights")
    self.assertShapeEquals((5, 7), fc_1)

    fc_2 = testing.get_variable_by_name(
        "time_feature_2_hidden/fully_connected_2/weights")
    self.assertShapeEquals((7, 7), fc_2)

    self.assertItemsEqual(["time_feature_1", "time_feature_2"],
                          model.time_series_hidden_layers.keys())
    self.assertShapeEquals((None, 20),
                           model.time_series_hidden_layers["time_feature_1"])
    self.assertShapeEquals((None, 7),
                           model.time_series_hidden_layers["time_feature_2"])
    self.assertItemsEqual(["aux_feature_1"], model.aux_hidden_layers.keys())
    self.assertIs(model.aux_features["aux_feature_1"],
                  model.aux_hidden_layers["aux_feature_1"])
    self.assertShapeEquals((None, 28), model.pre_logits_concat)

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

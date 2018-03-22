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

"""Tests for astro_model.AstroModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from astronet.astro_model import astro_model
from astronet.astro_model import configurations
from astronet.ops import input_ops
from astronet.ops import testing
from astronet.util import configdict


class AstroModelTest(tf.test.TestCase):

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

  def testInvalidModeRaisesError(self):
    # Build config.
    config = configdict.ConfigDict(configurations.base())

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    with self.assertRaises(ValueError):
      _ = astro_model.AstroModel(features, labels, config.hparams, "training")

  def testZeroFeaturesRaisesError(self):
    # Build config.
    config = configurations.base()
    config["inputs"]["features"] = {}
    config = configdict.ConfigDict(config)

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_model.AstroModel(features, labels, config.hparams,
                                   tf.estimator.ModeKeys.TRAIN)
    with self.assertRaises(ValueError):
      # Raises ValueError because at least one feature is required.
      model.build()

  def testOneTimeSeriesFeature(self):
    # Build config.
    feature_spec = {
        "time_feature_1": {
            "length": 10,
            "is_time_series": True,
        }
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config = configdict.ConfigDict(config)

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_model.AstroModel(features, labels, config.hparams,
                                   tf.estimator.ModeKeys.TRAIN)
    model.build()

    # Validate hidden layers.
    self.assertItemsEqual(["time_feature_1"],
                          model.time_series_hidden_layers.keys())
    self.assertIs(model.time_series_features["time_feature_1"],
                  model.time_series_hidden_layers["time_feature_1"])
    self.assertEqual(len(model.aux_hidden_layers), 0)
    self.assertIs(model.time_series_features["time_feature_1"],
                  model.pre_logits_concat)

  def testOneAuxFeature(self):
    # Build config.
    feature_spec = {
        "aux_feature_1": {
            "length": 1,
            "is_time_series": False,
        }
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config = configdict.ConfigDict(config)

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_model.AstroModel(features, labels, config.hparams,
                                   tf.estimator.ModeKeys.TRAIN)
    model.build()

    # Validate hidden layers.
    self.assertEqual(len(model.time_series_hidden_layers), 0)
    self.assertItemsEqual(["aux_feature_1"], model.aux_hidden_layers.keys())
    self.assertIs(model.aux_features["aux_feature_1"],
                  model.aux_hidden_layers["aux_feature_1"])
    self.assertIs(model.aux_features["aux_feature_1"], model.pre_logits_concat)

  def testTwoTimeSeriesFeatures(self):
    # Build config.
    feature_spec = {
        "time_feature_1": {
            "length": 10,
            "is_time_series": True,
        },
        "time_feature_2": {
            "length": 10,
            "is_time_series": True,
        }
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config = configdict.ConfigDict(config)

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_model.AstroModel(features, labels, config.hparams,
                                   tf.estimator.ModeKeys.TRAIN)
    model.build()

    # Validate hidden layers.
    self.assertItemsEqual(["time_feature_1", "time_feature_2"],
                          model.time_series_hidden_layers.keys())
    self.assertIs(model.time_series_features["time_feature_1"],
                  model.time_series_hidden_layers["time_feature_1"])
    self.assertIs(model.time_series_features["time_feature_2"],
                  model.time_series_hidden_layers["time_feature_2"])
    self.assertEqual(len(model.aux_hidden_layers), 0)
    self.assertShapeEquals((None, 20), model.pre_logits_concat)

  def testOneTimeSeriesOneAuxFeature(self):
    # Build config.
    feature_spec = {
        "time_feature_1": {
            "length": 10,
            "is_time_series": True,
        },
        "aux_feature_1": {
            "length": 1,
            "is_time_series": False,
        },
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config = configdict.ConfigDict(config)

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_model.AstroModel(features, labels, config.hparams,
                                   tf.estimator.ModeKeys.TRAIN)
    model.build()

    # Validate hidden layers.
    self.assertItemsEqual(["time_feature_1"],
                          model.time_series_hidden_layers.keys())
    self.assertIs(model.time_series_features["time_feature_1"],
                  model.time_series_hidden_layers["time_feature_1"])
    self.assertItemsEqual(["aux_feature_1"], model.aux_hidden_layers.keys())
    self.assertIs(model.aux_features["aux_feature_1"],
                  model.aux_hidden_layers["aux_feature_1"])
    self.assertShapeEquals((None, 11), model.pre_logits_concat)

  def testOneTimeSeriesTwoAuxFeatures(self):
    # Build config.
    feature_spec = {
        "time_feature_1": {
            "length": 10,
            "is_time_series": True,
        },
        "aux_feature_1": {
            "length": 1,
            "is_time_series": False,
        },
        "aux_feature_2": {
            "length": 2,
            "is_time_series": False,
        },
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config = configdict.ConfigDict(config)

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_model.AstroModel(features, labels, config.hparams,
                                   tf.estimator.ModeKeys.TRAIN)
    model.build()

    # Validate hidden layers.
    self.assertItemsEqual(["time_feature_1"],
                          model.time_series_hidden_layers.keys())
    self.assertIs(model.time_series_features["time_feature_1"],
                  model.time_series_hidden_layers["time_feature_1"])
    self.assertItemsEqual(["aux_feature_1", "aux_feature_2"],
                          model.aux_hidden_layers.keys())
    self.assertIs(model.aux_features["aux_feature_1"],
                  model.aux_hidden_layers["aux_feature_1"])
    self.assertIs(model.aux_features["aux_feature_2"],
                  model.aux_hidden_layers["aux_feature_2"])
    self.assertShapeEquals((None, 13), model.pre_logits_concat)

  def testZeroHiddenLayers(self):
    # Build config.
    feature_spec = {
        "time_feature_1": {
            "length": 10,
            "is_time_series": True,
        },
        "time_feature_2": {
            "length": 10,
            "is_time_series": True,
        },
        "aux_feature_1": {
            "length": 1,
            "is_time_series": False,
        },
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config = configdict.ConfigDict(config)
    config.hparams.output_dim = 1
    config.hparams.num_pre_logits_hidden_layers = 0

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_model.AstroModel(features, labels, config.hparams,
                                   tf.estimator.ModeKeys.TRAIN)
    model.build()

    # Validate Tensor shapes.
    self.assertShapeEquals((None, 21), model.pre_logits_concat)
    logits_w = testing.get_variable_by_name("logits/kernel")
    self.assertShapeEquals((21, 1), logits_w)

  def testOneHiddenLayer(self):
    # Build config.
    feature_spec = {
        "time_feature_1": {
            "length": 10,
            "is_time_series": True,
        },
        "time_feature_2": {
            "length": 10,
            "is_time_series": True,
        },
        "aux_feature_1": {
            "length": 1,
            "is_time_series": False,
        },
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config = configdict.ConfigDict(config)
    config.hparams.output_dim = 1
    config.hparams.num_pre_logits_hidden_layers = 1
    config.hparams.pre_logits_hidden_layer_size = 5

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_model.AstroModel(features, labels, config.hparams,
                                   tf.estimator.ModeKeys.TRAIN)
    model.build()

    # Validate Tensor shapes.
    self.assertShapeEquals((None, 21), model.pre_logits_concat)
    fc1 = testing.get_variable_by_name(
        "pre_logits_hidden/fully_connected_1/kernel")
    self.assertShapeEquals((21, 5), fc1)
    logits_w = testing.get_variable_by_name("logits/kernel")
    self.assertShapeEquals((5, 1), logits_w)

  def testTwoHiddenLayers(self):
    # Build config.
    feature_spec = {
        "time_feature_1": {
            "length": 10,
            "is_time_series": True,
        },
        "time_feature_2": {
            "length": 10,
            "is_time_series": True,
        },
        "aux_feature_1": {
            "length": 1,
            "is_time_series": False,
        },
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config = configdict.ConfigDict(config)
    config.hparams.output_dim = 1
    config.hparams.num_pre_logits_hidden_layers = 2
    config.hparams.pre_logits_hidden_layer_size = 5

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_model.AstroModel(features, labels, config.hparams,
                                   tf.estimator.ModeKeys.TRAIN)
    model.build()

    # Validate Tensor shapes.
    self.assertShapeEquals((None, 21), model.pre_logits_concat)
    fc1 = testing.get_variable_by_name(
        "pre_logits_hidden/fully_connected_1/kernel")
    self.assertShapeEquals((21, 5), fc1)
    fc2 = testing.get_variable_by_name(
        "pre_logits_hidden/fully_connected_2/kernel")
    self.assertShapeEquals((5, 5), fc2)
    logits_w = testing.get_variable_by_name("logits/kernel")
    self.assertShapeEquals((5, 1), logits_w)

  def testBinaryClassification(self):
    # Build config.
    feature_spec = {
        "time_feature_1": {
            "length": 10,
            "is_time_series": True,
        },
        "time_feature_2": {
            "length": 10,
            "is_time_series": True,
        },
        "aux_feature_1": {
            "length": 1,
            "is_time_series": False,
        },
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config = configdict.ConfigDict(config)
    config.hparams.output_dim = 1

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_model.AstroModel(features, labels, config.hparams,
                                   tf.estimator.ModeKeys.TRAIN)
    model.build()

    # Validate Tensor shapes.
    self.assertShapeEquals((None, 1), model.logits)
    self.assertShapeEquals((None, 1), model.predictions)
    self.assertShapeEquals((None,), model.batch_losses)
    self.assertShapeEquals((), model.total_loss)

    # Execute the TensorFlow graph.
    scaffold = tf.train.Scaffold()
    scaffold.finalize()
    with self.test_session() as sess:
      sess.run([scaffold.init_op, scaffold.local_init_op])
      step = sess.run(model.global_step)
      self.assertEqual(0, step)

      # Fetch total loss.
      features = testing.fake_features(feature_spec, batch_size=16)
      labels = testing.fake_labels(config.hparams.output_dim, batch_size=16)
      feed_dict = input_ops.prepare_feed_dict(model, features, labels)
      total_loss = sess.run(model.total_loss, feed_dict=feed_dict)
      self.assertShapeEquals((), total_loss)

  def testMultiLabelClassification(self):
    # Build config.
    feature_spec = {
        "time_feature_1": {
            "length": 10,
            "is_time_series": True,
        },
        "time_feature_2": {
            "length": 10,
            "is_time_series": True,
        },
        "aux_feature_1": {
            "length": 1,
            "is_time_series": False,
        },
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config = configdict.ConfigDict(config)
    config.hparams.output_dim = 3

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_model.AstroModel(features, labels, config.hparams,
                                   tf.estimator.ModeKeys.TRAIN)
    model.build()

    # Validate Tensor shapes.
    self.assertShapeEquals((None, 3), model.logits)
    self.assertShapeEquals((None, 3), model.predictions)
    self.assertShapeEquals((None,), model.batch_losses)
    self.assertShapeEquals((), model.total_loss)

    # Execute the TensorFlow graph.
    scaffold = tf.train.Scaffold()
    scaffold.finalize()
    with self.test_session() as sess:
      sess.run([scaffold.init_op, scaffold.local_init_op])
      step = sess.run(model.global_step)
      self.assertEqual(0, step)

      # Fetch total loss.
      features = testing.fake_features(feature_spec, batch_size=16)
      labels = testing.fake_labels(config.hparams.output_dim, batch_size=16)
      feed_dict = input_ops.prepare_feed_dict(model, features, labels)
      total_loss = sess.run(model.total_loss, feed_dict=feed_dict)
      self.assertShapeEquals((), total_loss)

  def testEvalMode(self):
    # Build config.
    feature_spec = {
        "time_feature_1": {
            "length": 10,
            "is_time_series": True,
        },
        "time_feature_2": {
            "length": 10,
            "is_time_series": True,
        },
        "aux_feature_1": {
            "length": 1,
            "is_time_series": False,
        },
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config = configdict.ConfigDict(config)
    config.hparams.output_dim = 1

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_model.AstroModel(features, labels, config.hparams,
                                   tf.estimator.ModeKeys.TRAIN)
    model.build()

    # Validate Tensor shapes.
    self.assertShapeEquals((None, 21), model.pre_logits_concat)
    self.assertShapeEquals((None, 1), model.logits)
    self.assertShapeEquals((None, 1), model.predictions)
    self.assertShapeEquals((None,), model.batch_losses)
    self.assertShapeEquals((), model.total_loss)

    # Execute the TensorFlow graph.
    scaffold = tf.train.Scaffold()
    scaffold.finalize()
    with self.test_session() as sess:
      sess.run([scaffold.init_op, scaffold.local_init_op])
      step = sess.run(model.global_step)
      self.assertEqual(0, step)

      # Fetch total loss.
      features = testing.fake_features(feature_spec, batch_size=16)
      labels = testing.fake_labels(config.hparams.output_dim, batch_size=16)
      feed_dict = input_ops.prepare_feed_dict(model, features, labels)
      total_loss = sess.run(model.total_loss, feed_dict=feed_dict)
      self.assertShapeEquals((), total_loss)

  def testPredictMode(self):
    # Build config.
    feature_spec = {
        "time_feature_1": {
            "length": 10,
            "is_time_series": True,
        },
        "time_feature_2": {
            "length": 10,
            "is_time_series": True,
        },
        "aux_feature_1": {
            "length": 1,
            "is_time_series": False,
        },
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config = configdict.ConfigDict(config)
    config.hparams.output_dim = 1

    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    model = astro_model.AstroModel(features, None, config.hparams,
                                   tf.estimator.ModeKeys.PREDICT)
    model.build()

    # Validate Tensor shapes.
    self.assertIsNone(model.labels)
    self.assertShapeEquals((None, 21), model.pre_logits_concat)
    self.assertShapeEquals((None, 1), model.logits)
    self.assertShapeEquals((None, 1), model.predictions)
    self.assertIsNone(model.batch_losses)
    self.assertIsNone(model.total_loss)

    # Execute the TensorFlow graph.
    scaffold = tf.train.Scaffold()
    scaffold.finalize()
    with self.test_session() as sess:
      sess.run([scaffold.init_op, scaffold.local_init_op])
      step = sess.run(model.global_step)
      self.assertEqual(0, step)

      # Fetch predictions.
      features = testing.fake_features(feature_spec, batch_size=16)
      feed_dict = input_ops.prepare_feed_dict(model, features)
      predictions = sess.run(model.predictions, feed_dict=feed_dict)
      self.assertShapeEquals((16, 1), predictions)


if __name__ == "__main__":
  tf.test.main()

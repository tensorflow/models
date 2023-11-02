# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for cls_head."""
from absl.testing import parameterized

import tensorflow as tf, tf_keras

from official.nlp.modeling.layers import cls_head


class ClassificationHeadTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("no_pooler_layer", 0, 2),
                                  ("has_pooler_layer", 5, 4))
  def test_pooler_layer(self, inner_dim, num_weights_expected):
    test_layer = cls_head.ClassificationHead(inner_dim=inner_dim, num_classes=2)
    features = tf.zeros(shape=(2, 10, 10), dtype=tf.float32)
    _ = test_layer(features)

    num_weights_observed = len(test_layer.get_weights())
    self.assertEqual(num_weights_observed, num_weights_expected)

  def test_layer_invocation(self):
    test_layer = cls_head.ClassificationHead(inner_dim=5, num_classes=2)
    features = tf.zeros(shape=(2, 10, 10), dtype=tf.float32)
    output = test_layer(features)
    self.assertAllClose(output, [[0., 0.], [0., 0.]])
    self.assertSameElements(test_layer.checkpoint_items.keys(),
                            ["pooler_dense"])
    outputs = test_layer(features, only_project=True)
    self.assertEqual(outputs.shape, (2, 5))

  def test_layer_serialization(self):
    layer = cls_head.ClassificationHead(10, 2)
    new_layer = cls_head.ClassificationHead.from_config(layer.get_config())

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(layer.get_config(), new_layer.get_config())


class MultiClsHeadsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("no_pooler_layer", 0, 4),
                                  ("has_pooler_layer", 5, 6))
  def test_pooler_layer(self, inner_dim, num_weights_expected):
    cls_list = [("foo", 2), ("bar", 3)]
    test_layer = cls_head.MultiClsHeads(inner_dim=inner_dim, cls_list=cls_list)
    features = tf.zeros(shape=(2, 10, 10), dtype=tf.float32)
    _ = test_layer(features)

    num_weights_observed = len(test_layer.get_weights())
    self.assertEqual(num_weights_observed, num_weights_expected)

  def test_layer_invocation(self):
    cls_list = [("foo", 2), ("bar", 3)]
    test_layer = cls_head.MultiClsHeads(inner_dim=5, cls_list=cls_list)
    features = tf.zeros(shape=(2, 10, 10), dtype=tf.float32)
    outputs = test_layer(features)
    self.assertAllClose(outputs["foo"], [[0., 0.], [0., 0.]])
    self.assertAllClose(outputs["bar"], [[0., 0., 0.], [0., 0., 0.]])
    self.assertSameElements(test_layer.checkpoint_items.keys(),
                            ["pooler_dense", "foo", "bar"])

    outputs = test_layer(features, only_project=True)
    self.assertEqual(outputs.shape, (2, 5))

  def test_layer_serialization(self):
    cls_list = [("foo", 2), ("bar", 3)]
    test_layer = cls_head.MultiClsHeads(inner_dim=5, cls_list=cls_list)
    new_layer = cls_head.MultiClsHeads.from_config(test_layer.get_config())

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(test_layer.get_config(), new_layer.get_config())


class GaussianProcessClassificationHead(tf.test.TestCase,
                                        parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.spec_norm_kwargs = dict(norm_multiplier=1.,)
    self.gp_layer_kwargs = dict(num_inducing=512)

  @parameterized.named_parameters(("no_pooler_layer", 0, 7),
                                  ("has_pooler_layer", 5, 11))
  def test_pooler_layer(self, inner_dim, num_weights_expected):
    test_layer = cls_head.GaussianProcessClassificationHead(
        inner_dim=inner_dim,
        num_classes=2,
        use_spec_norm=True,
        use_gp_layer=True,
        initializer="zeros",
        **self.spec_norm_kwargs,
        **self.gp_layer_kwargs)
    features = tf.zeros(shape=(2, 10, 10), dtype=tf.float32)
    _ = test_layer(features)

    num_weights_observed = len(test_layer.get_weights())
    self.assertEqual(num_weights_observed, num_weights_expected)

  def test_layer_invocation(self):
    test_layer = cls_head.GaussianProcessClassificationHead(
        inner_dim=5,
        num_classes=2,
        use_spec_norm=True,
        use_gp_layer=True,
        initializer="zeros",
        **self.spec_norm_kwargs,
        **self.gp_layer_kwargs)
    features = tf.zeros(shape=(2, 10, 10), dtype=tf.float32)
    output = test_layer(features)
    self.assertAllClose(output, [[0., 0.], [0., 0.]])
    self.assertSameElements(test_layer.checkpoint_items.keys(),
                            ["pooler_dense"])

  @parameterized.named_parameters(
      ("gp_layer_with_covmat", True, True),
      ("gp_layer_no_covmat", True, False),
      ("dense_layer_with_covmat", False, True),
      ("dense_layer_no_covmat", False, False))
  def test_sngp_output_shape(self, use_gp_layer, return_covmat):
    batch_size = 32
    num_classes = 2

    test_layer = cls_head.GaussianProcessClassificationHead(
        inner_dim=5,
        num_classes=num_classes,
        use_spec_norm=True,
        use_gp_layer=use_gp_layer,
        **self.spec_norm_kwargs,
        **self.gp_layer_kwargs)

    features = tf.zeros(shape=(batch_size, 10, 10), dtype=tf.float32)
    outputs = test_layer(features, return_covmat=return_covmat)

    if use_gp_layer and return_covmat:
      self.assertIsInstance(outputs, tuple)
      self.assertEqual(outputs[0].shape, (batch_size, num_classes))
      self.assertEqual(outputs[1].shape, (batch_size, batch_size))
    else:
      self.assertIsInstance(outputs, tf.Tensor)
      self.assertEqual(outputs.shape, (batch_size, num_classes))

  def test_sngp_train_logits(self):
    """Checks if temperature scaling is disabled during training."""
    features = tf.zeros(shape=(5, 10, 10), dtype=tf.float32)

    gp_layer = cls_head.GaussianProcessClassificationHead(
        inner_dim=5, num_classes=2)

    # Without temperature.
    gp_layer.temperature = None
    outputs_no_temp = gp_layer(features, training=True)

    # With temperature.
    gp_layer.temperature = 10.
    outputs_with_temp = gp_layer(features, training=True)

    self.assertAllEqual(outputs_no_temp, outputs_with_temp)

  def test_layer_serialization(self):
    layer = cls_head.GaussianProcessClassificationHead(
        inner_dim=5,
        num_classes=2,
        use_spec_norm=True,
        use_gp_layer=True,
        **self.spec_norm_kwargs,
        **self.gp_layer_kwargs)
    new_layer = cls_head.GaussianProcessClassificationHead.from_config(
        layer.get_config())

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(layer.get_config(), new_layer.get_config())

  def test_sngp_kwargs_serialization(self):
    """Tests if SNGP-specific kwargs are added during serialization."""
    layer = cls_head.GaussianProcessClassificationHead(
        inner_dim=5,
        num_classes=2,
        use_spec_norm=True,
        use_gp_layer=True,
        **self.spec_norm_kwargs,
        **self.gp_layer_kwargs)
    layer_config = layer.get_config()

    # The config value should equal to those defined in setUp().
    self.assertEqual(layer_config["norm_multiplier"], 1.)
    self.assertEqual(layer_config["num_inducing"], 512)


class PerQueryDenseHeadTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("single_query", 1, 3, False),
                                  ("multi_queries", 10, 2, False),
                                  ("with_bias", 10, 2, True))
  def test_layer_invocation(self, num_queries, features, use_bias):
    batch_size = 5
    hidden_size = 10
    layer = cls_head.PerQueryDenseHead(
        num_queries=num_queries, features=features, use_bias=use_bias)
    inputs = tf.zeros(
        shape=(batch_size, num_queries, hidden_size), dtype=tf.float32)
    outputs = layer(inputs)
    self.assertEqual(outputs.shape, [batch_size, num_queries, features])

  def test_layer_serialization(self):
    layer = cls_head.PerQueryDenseHead(
        num_queries=10, features=2, use_bias=True)
    new_layer = cls_head.PerQueryDenseHead.from_config(layer.get_config())

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(layer.get_config(), new_layer.get_config())

if __name__ == "__main__":
  tf.test.main()

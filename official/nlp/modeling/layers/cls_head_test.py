# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf

from official.nlp.modeling.layers import cls_head


class ClassificationHeadTest(tf.test.TestCase):

  def test_layer_invocation(self):
    test_layer = cls_head.ClassificationHead(inner_dim=5, num_classes=2)
    features = tf.zeros(shape=(2, 10, 10), dtype=tf.float32)
    output = test_layer(features)
    self.assertAllClose(output, [[0., 0.], [0., 0.]])
    self.assertSameElements(test_layer.checkpoint_items.keys(),
                            ["pooler_dense"])

  def test_layer_serialization(self):
    layer = cls_head.ClassificationHead(10, 2)
    new_layer = cls_head.ClassificationHead.from_config(layer.get_config())

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(layer.get_config(), new_layer.get_config())


class MultiClsHeadsTest(tf.test.TestCase):

  def test_layer_invocation(self):
    cls_list = [("foo", 2), ("bar", 3)]
    test_layer = cls_head.MultiClsHeads(inner_dim=5, cls_list=cls_list)
    features = tf.zeros(shape=(2, 10, 10), dtype=tf.float32)
    outputs = test_layer(features)
    self.assertAllClose(outputs["foo"], [[0., 0.], [0., 0.]])
    self.assertAllClose(outputs["bar"], [[0., 0., 0.], [0., 0., 0.]])
    self.assertSameElements(test_layer.checkpoint_items.keys(),
                            ["pooler_dense", "foo", "bar"])

  def test_layer_serialization(self):
    cls_list = [("foo", 2), ("bar", 3)]
    test_layer = cls_head.MultiClsHeads(inner_dim=5, cls_list=cls_list)
    new_layer = cls_head.MultiClsHeads.from_config(test_layer.get_config())

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(test_layer.get_config(), new_layer.get_config())


class GaussianProcessClassificationHead(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.spec_norm_kwargs = dict(norm_multiplier=1.,)
    self.gp_layer_kwargs = dict(num_inducing=512)

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
    output, _ = test_layer(features)
    self.assertAllClose(output, [[0., 0.], [0., 0.]])
    self.assertSameElements(test_layer.checkpoint_items.keys(),
                            ["pooler_dense"])

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

if __name__ == "__main__":
  tf.test.main()

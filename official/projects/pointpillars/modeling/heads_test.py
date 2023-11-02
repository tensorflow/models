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

"""Tests for decoders."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.pointpillars.modeling import heads


class SSDHeadTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (2, [], 1, 1),
      (3, [{'name': 'z', 'type': 'regression', 'size': 1}], 1, 3))
  def test_network_creation(self, num_classes, attribute_heads, min_level,
                            max_level):
    """Test if network could be created and infer with expected shapes."""
    # Fix the input shape, anchor size and num of conv filters.
    n, h, w, c = 1, 32, 32, 4
    num_anchors_per_location = 3
    num_params_per_anchor = 4
    inputs = {'1': tf_keras.Input(shape=[h, w, c], batch_size=n)}

    head = heads.SSDHead(num_classes, num_anchors_per_location,
                         num_params_per_anchor, attribute_heads, min_level,
                         max_level)
    scores, boxes, attributes = head(inputs)
    for level in range(min_level, max_level+1):
      self.assertIn(str(level), scores)
      self.assertIn(str(level), boxes)
      scale = 2**(level - min_level)
      self.assertAllEqual(scores[str(level)].shape.as_list(), [
          n,
          int(h / scale),
          int(w / scale), num_classes * num_anchors_per_location
      ])
      self.assertAllEqual(boxes[str(level)].shape.as_list(), [
          n,
          int(h / scale),
          int(w / scale), num_params_per_anchor * num_anchors_per_location
      ])
    for attr_head in attribute_heads:
      name = attr_head['name']
      size = attr_head['size']
      self.assertIn(name, attributes)
      attr = attributes[name]
      for level in range(min_level, max_level+1):
        self.assertIn(str(level), attr)
        scale = 2**(level - min_level)
        self.assertAllEqual(attr[str(level)].shape.as_list(), [
            n,
            int(h / scale),
            int(w / scale), size * num_anchors_per_location
        ])

  def test_serialization(self):
    kwargs = dict(
        num_classes=2,
        num_anchors_per_location=3,
        num_params_per_anchor=4,
        attribute_heads=[
            {'name': 'z', 'type': 'regression', 'size': 1},
        ],
        min_level=1,
        max_level=3,
        kernel_regularizer=None
    )
    net = heads.SSDHead(**kwargs)
    expected_config = kwargs
    self.assertEqual(net.get_config(), expected_config)

    new_net = heads.SSDHead.from_config(net.get_config())
    self.assertAllEqual(net.get_config(), new_net.get_config())


if __name__ == '__main__':
  tf.test.main()

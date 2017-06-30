# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for feature map generators."""

import tensorflow as tf

from object_detection.models import feature_map_generators

INCEPTION_V2_LAYOUT = {
    'from_layer': ['Mixed_3c', 'Mixed_4c', 'Mixed_5c', '', '', ''],
    'layer_depth': [-1, -1, -1, 512, 256, 256],
    'anchor_strides': [16, 32, 64, -1, -1, -1],
    'layer_target_norm': [20.0, -1, -1, -1, -1, -1],
}

INCEPTION_V3_LAYOUT = {
    'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''],
    'layer_depth': [-1, -1, -1, 512, 256, 128],
    'anchor_strides': [16, 32, 64, -1, -1, -1],
    'aspect_ratios': [1.0, 2.0, 1.0/2, 3.0, 1.0/3]
}


# TODO: add tests with different anchor strides.
class MultiResolutionFeatureMapGeneratorTest(tf.test.TestCase):

  def test_get_expected_feature_map_shapes_with_inception_v2(self):
    image_features = {
        'Mixed_3c': tf.random_uniform([4, 28, 28, 256], dtype=tf.float32),
        'Mixed_4c': tf.random_uniform([4, 14, 14, 576], dtype=tf.float32),
        'Mixed_5c': tf.random_uniform([4, 7, 7, 1024], dtype=tf.float32)
    }
    feature_maps = feature_map_generators.multi_resolution_feature_maps(
        feature_map_layout=INCEPTION_V2_LAYOUT,
        depth_multiplier=1,
        min_depth=32,
        insert_1x1_conv=True,
        image_features=image_features)

    expected_feature_map_shapes = {
        'Mixed_3c': (4, 28, 28, 256),
        'Mixed_4c': (4, 14, 14, 576),
        'Mixed_5c': (4, 7, 7, 1024),
        'Mixed_5c_2_Conv2d_3_3x3_s2_512': (4, 4, 4, 512),
        'Mixed_5c_2_Conv2d_4_3x3_s2_256': (4, 2, 2, 256),
        'Mixed_5c_2_Conv2d_5_3x3_s2_256': (4, 1, 1, 256)}

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      out_feature_maps = sess.run(feature_maps)
      out_feature_map_shapes = dict(
          (key, value.shape) for key, value in out_feature_maps.iteritems())
      self.assertDictEqual(out_feature_map_shapes, expected_feature_map_shapes)

  def test_get_expected_feature_map_shapes_with_inception_v3(self):
    image_features = {
        'Mixed_5d': tf.random_uniform([4, 35, 35, 256], dtype=tf.float32),
        'Mixed_6e': tf.random_uniform([4, 17, 17, 576], dtype=tf.float32),
        'Mixed_7c': tf.random_uniform([4, 8, 8, 1024], dtype=tf.float32)
    }

    feature_maps = feature_map_generators.multi_resolution_feature_maps(
        feature_map_layout=INCEPTION_V3_LAYOUT,
        depth_multiplier=1,
        min_depth=32,
        insert_1x1_conv=True,
        image_features=image_features)

    expected_feature_map_shapes = {
        'Mixed_5d': (4, 35, 35, 256),
        'Mixed_6e': (4, 17, 17, 576),
        'Mixed_7c': (4, 8, 8, 1024),
        'Mixed_7c_2_Conv2d_3_3x3_s2_512': (4, 4, 4, 512),
        'Mixed_7c_2_Conv2d_4_3x3_s2_256': (4, 2, 2, 256),
        'Mixed_7c_2_Conv2d_5_3x3_s2_128': (4, 1, 1, 128)}

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      out_feature_maps = sess.run(feature_maps)
      out_feature_map_shapes = dict(
          (key, value.shape) for key, value in out_feature_maps.iteritems())
      self.assertDictEqual(out_feature_map_shapes, expected_feature_map_shapes)


class GetDepthFunctionTest(tf.test.TestCase):

  def test_return_min_depth_when_multiplier_is_small(self):
    depth_fn = feature_map_generators.get_depth_fn(depth_multiplier=0.5,
                                                   min_depth=16)
    self.assertEqual(depth_fn(16), 16)

  def test_return_correct_depth_with_multiplier(self):
    depth_fn = feature_map_generators.get_depth_fn(depth_multiplier=0.5,
                                                   min_depth=16)
    self.assertEqual(depth_fn(64), 32)


if __name__ == '__main__':
  tf.test.main()

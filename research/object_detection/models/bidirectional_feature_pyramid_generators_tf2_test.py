# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for bidirectional feature pyramid generators."""
import unittest
from absl.testing import parameterized

import tensorflow.compat.v1 as tf

from google.protobuf import text_format

from object_detection.builders import hyperparams_builder
from object_detection.models import bidirectional_feature_pyramid_generators as bifpn_generators
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case
from object_detection.utils import test_utils
from object_detection.utils import tf_version


@parameterized.parameters({'bifpn_num_iterations': 2},
                          {'bifpn_num_iterations': 8})
@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class BiFPNFeaturePyramidGeneratorTest(test_case.TestCase):

  def _build_conv_hyperparams(self):
    conv_hyperparams = hyperparams_pb2.Hyperparams()
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      force_use_bias: true
    """
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
    return hyperparams_builder.KerasLayerHyperparams(conv_hyperparams)

  def test_get_expected_feature_map_shapes(self, bifpn_num_iterations):
    with test_utils.GraphContextOrNone() as g:
      image_features = [
          ('block3', tf.random_uniform([4, 16, 16, 256], dtype=tf.float32)),
          ('block4', tf.random_uniform([4, 8, 8, 256], dtype=tf.float32)),
          ('block5', tf.random_uniform([4, 4, 4, 256], dtype=tf.float32))
      ]
      bifpn_generator = bifpn_generators.KerasBiFpnFeatureMaps(
          bifpn_num_iterations=bifpn_num_iterations,
          bifpn_num_filters=128,
          fpn_min_level=3,
          fpn_max_level=7,
          input_max_level=5,
          is_training=True,
          conv_hyperparams=self._build_conv_hyperparams(),
          freeze_batchnorm=False)
    def graph_fn():
      feature_maps = bifpn_generator(image_features)
      return feature_maps

    expected_feature_map_shapes = {
        '{}_dn_lvl_3'.format(bifpn_num_iterations): (4, 16, 16, 128),
        '{}_up_lvl_4'.format(bifpn_num_iterations): (4, 8, 8, 128),
        '{}_up_lvl_5'.format(bifpn_num_iterations): (4, 4, 4, 128),
        '{}_up_lvl_6'.format(bifpn_num_iterations): (4, 2, 2, 128),
        '{}_up_lvl_7'.format(bifpn_num_iterations): (4, 1, 1, 128)}
    out_feature_maps = self.execute(graph_fn, [], g)
    out_feature_map_shapes = dict(
        (key, value.shape) for key, value in out_feature_maps.items())
    self.assertDictEqual(expected_feature_map_shapes, out_feature_map_shapes)

  def test_get_expected_variable_names(self, bifpn_num_iterations):
    with test_utils.GraphContextOrNone() as g:
      image_features = [
          ('block3', tf.random_uniform([4, 16, 16, 256], dtype=tf.float32)),
          ('block4', tf.random_uniform([4, 8, 8, 256], dtype=tf.float32)),
          ('block5', tf.random_uniform([4, 4, 4, 256], dtype=tf.float32))
      ]
      bifpn_generator = bifpn_generators.KerasBiFpnFeatureMaps(
          bifpn_num_iterations=bifpn_num_iterations,
          bifpn_num_filters=128,
          fpn_min_level=3,
          fpn_max_level=7,
          input_max_level=5,
          is_training=True,
          conv_hyperparams=self._build_conv_hyperparams(),
          freeze_batchnorm=False,
          name='bifpn')
    def graph_fn():
      return bifpn_generator(image_features)

    self.execute(graph_fn, [], g)
    expected_variables = [
        'bifpn/node_00/0_up_lvl_6/input_0_up_lvl_5/1x1_pre_sample/conv/bias',
        'bifpn/node_00/0_up_lvl_6/input_0_up_lvl_5/1x1_pre_sample/conv/kernel',
        'bifpn/node_03/1_dn_lvl_5/input_0_up_lvl_5/1x1_pre_sample/conv/bias',
        'bifpn/node_03/1_dn_lvl_5/input_0_up_lvl_5/1x1_pre_sample/conv/kernel',
        'bifpn/node_04/1_dn_lvl_4/input_0_up_lvl_4/1x1_pre_sample/conv/bias',
        'bifpn/node_04/1_dn_lvl_4/input_0_up_lvl_4/1x1_pre_sample/conv/kernel',
        'bifpn/node_05/1_dn_lvl_3/input_0_up_lvl_3/1x1_pre_sample/conv/bias',
        'bifpn/node_05/1_dn_lvl_3/input_0_up_lvl_3/1x1_pre_sample/conv/kernel',
        'bifpn/node_06/1_up_lvl_4/input_0_up_lvl_4/1x1_pre_sample/conv/bias',
        'bifpn/node_06/1_up_lvl_4/input_0_up_lvl_4/1x1_pre_sample/conv/kernel',
        'bifpn/node_07/1_up_lvl_5/input_0_up_lvl_5/1x1_pre_sample/conv/bias',
        'bifpn/node_07/1_up_lvl_5/input_0_up_lvl_5/1x1_pre_sample/conv/kernel']
    expected_node_variable_patterns = [
        ['bifpn/node_{:02}/{}_dn_lvl_6/combine/bifpn_combine_weights',
         'bifpn/node_{:02}/{}_dn_lvl_6/post_combine/separable_conv/bias',
         'bifpn/node_{:02}/{}_dn_lvl_6/post_combine/separable_conv/depthwise_kernel',
         'bifpn/node_{:02}/{}_dn_lvl_6/post_combine/separable_conv/pointwise_kernel'],
        ['bifpn/node_{:02}/{}_dn_lvl_5/combine/bifpn_combine_weights',
         'bifpn/node_{:02}/{}_dn_lvl_5/post_combine/separable_conv/bias',
         'bifpn/node_{:02}/{}_dn_lvl_5/post_combine/separable_conv/depthwise_kernel',
         'bifpn/node_{:02}/{}_dn_lvl_5/post_combine/separable_conv/pointwise_kernel'],
        ['bifpn/node_{:02}/{}_dn_lvl_4/combine/bifpn_combine_weights',
         'bifpn/node_{:02}/{}_dn_lvl_4/post_combine/separable_conv/bias',
         'bifpn/node_{:02}/{}_dn_lvl_4/post_combine/separable_conv/depthwise_kernel',
         'bifpn/node_{:02}/{}_dn_lvl_4/post_combine/separable_conv/pointwise_kernel'],
        ['bifpn/node_{:02}/{}_dn_lvl_3/combine/bifpn_combine_weights',
         'bifpn/node_{:02}/{}_dn_lvl_3/post_combine/separable_conv/bias',
         'bifpn/node_{:02}/{}_dn_lvl_3/post_combine/separable_conv/depthwise_kernel',
         'bifpn/node_{:02}/{}_dn_lvl_3/post_combine/separable_conv/pointwise_kernel'],
        ['bifpn/node_{:02}/{}_up_lvl_4/combine/bifpn_combine_weights',
         'bifpn/node_{:02}/{}_up_lvl_4/post_combine/separable_conv/bias',
         'bifpn/node_{:02}/{}_up_lvl_4/post_combine/separable_conv/depthwise_kernel',
         'bifpn/node_{:02}/{}_up_lvl_4/post_combine/separable_conv/pointwise_kernel'],
        ['bifpn/node_{:02}/{}_up_lvl_5/combine/bifpn_combine_weights',
         'bifpn/node_{:02}/{}_up_lvl_5/post_combine/separable_conv/bias',
         'bifpn/node_{:02}/{}_up_lvl_5/post_combine/separable_conv/depthwise_kernel',
         'bifpn/node_{:02}/{}_up_lvl_5/post_combine/separable_conv/pointwise_kernel'],
        ['bifpn/node_{:02}/{}_up_lvl_6/combine/bifpn_combine_weights',
         'bifpn/node_{:02}/{}_up_lvl_6/post_combine/separable_conv/bias',
         'bifpn/node_{:02}/{}_up_lvl_6/post_combine/separable_conv/depthwise_kernel',
         'bifpn/node_{:02}/{}_up_lvl_6/post_combine/separable_conv/pointwise_kernel'],
        ['bifpn/node_{:02}/{}_up_lvl_7/combine/bifpn_combine_weights',
         'bifpn/node_{:02}/{}_up_lvl_7/post_combine/separable_conv/bias',
         'bifpn/node_{:02}/{}_up_lvl_7/post_combine/separable_conv/depthwise_kernel',
         'bifpn/node_{:02}/{}_up_lvl_7/post_combine/separable_conv/pointwise_kernel']]

    node_i = 2
    for iter_i in range(1, bifpn_num_iterations+1):
      for node_variable_patterns in expected_node_variable_patterns:
        for pattern in node_variable_patterns:
          expected_variables.append(pattern.format(node_i, iter_i))
        node_i += 1

    expected_variables = set(expected_variables)
    actual_variable_set = set(
        [var.name.split(':')[0] for var in bifpn_generator.variables])
    self.assertSetEqual(expected_variables, actual_variable_set)

# TODO(aom): Tests for create_bifpn_combine_op.

if __name__ == '__main__':
  tf.test.main()

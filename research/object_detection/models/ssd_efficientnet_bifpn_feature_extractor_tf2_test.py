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
"""Tests for the ssd_efficientnet_bifpn_feature_extractor."""
import unittest
from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

from google.protobuf import text_format
from object_detection.builders import hyperparams_builder
from object_detection.models import ssd_efficientnet_bifpn_feature_extractor
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case
from object_detection.utils import tf_version


def _count_params(model, trainable_only=True):
  """Returns the count of all model parameters, or just trainable ones."""
  if not trainable_only:
    return model.count_params()
  else:
    return int(np.sum([
        tf.keras.backend.count_params(p) for p in model.trainable_weights]))


@parameterized.parameters(
    {'efficientdet_version': 'efficientdet-d0',
     'efficientnet_version': 'efficientnet-b0',
     'bifpn_num_iterations': 3,
     'bifpn_num_filters': 64,
     'bifpn_combine_method': 'fast_attention'},
    {'efficientdet_version': 'efficientdet-d1',
     'efficientnet_version': 'efficientnet-b1',
     'bifpn_num_iterations': 4,
     'bifpn_num_filters': 88,
     'bifpn_combine_method': 'fast_attention'},
    {'efficientdet_version': 'efficientdet-d2',
     'efficientnet_version': 'efficientnet-b2',
     'bifpn_num_iterations': 5,
     'bifpn_num_filters': 112,
     'bifpn_combine_method': 'fast_attention'},
    {'efficientdet_version': 'efficientdet-d3',
     'efficientnet_version': 'efficientnet-b3',
     'bifpn_num_iterations': 6,
     'bifpn_num_filters': 160,
     'bifpn_combine_method': 'fast_attention'},
    {'efficientdet_version': 'efficientdet-d4',
     'efficientnet_version': 'efficientnet-b4',
     'bifpn_num_iterations': 7,
     'bifpn_num_filters': 224,
     'bifpn_combine_method': 'fast_attention'},
    {'efficientdet_version': 'efficientdet-d5',
     'efficientnet_version': 'efficientnet-b5',
     'bifpn_num_iterations': 7,
     'bifpn_num_filters': 288,
     'bifpn_combine_method': 'fast_attention'},
    # efficientdet-d6 and efficientdet-d7 only differ in input size.
    {'efficientdet_version': 'efficientdet-d6-d7',
     'efficientnet_version': 'efficientnet-b6',
     'bifpn_num_iterations': 8,
     'bifpn_num_filters': 384,
     'bifpn_combine_method': 'sum'})
@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class SSDEfficientNetBiFPNFeatureExtractorTest(
    test_case.TestCase, parameterized.TestCase):

  def _build_conv_hyperparams(self, add_batch_norm=True):
    conv_hyperparams = hyperparams_pb2.Hyperparams()
    conv_hyperparams_text_proto = """
      force_use_bias: true
      activation: SWISH
      regularizer {
        l2_regularizer {
          weight: 0.0004
        }
      }
      initializer {
        truncated_normal_initializer {
          stddev: 0.03
          mean: 0.0
        }
      }
    """
    if add_batch_norm:
      batch_norm_proto = """
        batch_norm {
          scale: true,
          decay: 0.99,
          epsilon: 0.001,
        }
      """
      conv_hyperparams_text_proto += batch_norm_proto
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
    return hyperparams_builder.KerasLayerHyperparams(conv_hyperparams)

  def _create_feature_extractor(self,
                                efficientnet_version='efficientnet-b0',
                                bifpn_num_iterations=3,
                                bifpn_num_filters=64,
                                bifpn_combine_method='fast_attention'):
    """Constructs a new EfficientNetBiFPN feature extractor."""
    depth_multiplier = 1.0
    pad_to_multiple = 1
    min_depth = 16
    return (ssd_efficientnet_bifpn_feature_extractor
            .SSDEfficientNetBiFPNKerasFeatureExtractor(
                is_training=True,
                depth_multiplier=depth_multiplier,
                min_depth=min_depth,
                pad_to_multiple=pad_to_multiple,
                conv_hyperparams=self._build_conv_hyperparams(),
                freeze_batchnorm=False,
                inplace_batchnorm_update=False,
                bifpn_min_level=3,
                bifpn_max_level=7,
                bifpn_num_iterations=bifpn_num_iterations,
                bifpn_num_filters=bifpn_num_filters,
                bifpn_combine_method=bifpn_combine_method,
                efficientnet_version=efficientnet_version))

  def test_efficientdet_feature_extractor_shapes(self,
                                                 efficientdet_version,
                                                 efficientnet_version,
                                                 bifpn_num_iterations,
                                                 bifpn_num_filters,
                                                 bifpn_combine_method):
    feature_extractor = self._create_feature_extractor(
        efficientnet_version=efficientnet_version,
        bifpn_num_iterations=bifpn_num_iterations,
        bifpn_num_filters=bifpn_num_filters,
        bifpn_combine_method=bifpn_combine_method)
    outputs = feature_extractor(np.zeros((2, 256, 256, 3), dtype=np.float32))

    self.assertEqual(outputs[0].shape, (2, 32, 32, bifpn_num_filters))
    self.assertEqual(outputs[1].shape, (2, 16, 16, bifpn_num_filters))
    self.assertEqual(outputs[2].shape, (2, 8, 8, bifpn_num_filters))
    self.assertEqual(outputs[3].shape, (2, 4, 4, bifpn_num_filters))
    self.assertEqual(outputs[4].shape, (2, 2, 2, bifpn_num_filters))

  def test_efficientdet_feature_extractor_params(self,
                                                 efficientdet_version,
                                                 efficientnet_version,
                                                 bifpn_num_iterations,
                                                 bifpn_num_filters,
                                                 bifpn_combine_method):
    feature_extractor = self._create_feature_extractor(
        efficientnet_version=efficientnet_version,
        bifpn_num_iterations=bifpn_num_iterations,
        bifpn_num_filters=bifpn_num_filters,
        bifpn_combine_method=bifpn_combine_method)
    _ = feature_extractor(np.zeros((2, 256, 256, 3), dtype=np.float32))
    expected_params = {
        'efficientdet-d0': 5484829,
        'efficientdet-d1': 8185156,
        'efficientdet-d2': 9818153,
        'efficientdet-d3': 13792706,
        'efficientdet-d4': 22691445,
        'efficientdet-d5': 35795677,
        'efficientdet-d6-d7': 53624512,
    }
    num_params = _count_params(feature_extractor)
    self.assertEqual(expected_params[efficientdet_version], num_params)


if __name__ == '__main__':
  tf.test.main()

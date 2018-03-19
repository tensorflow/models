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

"""Tests for input_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from astronet.ops import input_ops
from astronet.util import configdict


class InputOpsTest(tf.test.TestCase):

  def assertFeatureShapesEqual(self, expected_shapes, features):
    """Asserts that a dict of feature placeholders has the expected shapes.

    Args:
      expected_shapes: Dictionary of expected Tensor shapes, as lists,
          corresponding to the structure of 'features'.
      features: Dictionary of feature placeholders of the format returned by
          input_ops.build_feature_placeholders().
    """
    actual_shapes = {}
    for feature_type in features:
      actual_shapes[feature_type] = {
          feature: tensor.shape.as_list()
          for feature, tensor in features[feature_type].items()
      }
    self.assertDictEqual(expected_shapes, actual_shapes)

  def testBuildFeaturePlaceholders(self):
    # One time series feature.
    config = configdict.ConfigDict({
        "time_feature_1": {
            "length": 14,
            "is_time_series": True,
        }
    })
    expected_shapes = {
        "time_series_features": {
            "time_feature_1": [None, 14],
        },
        "aux_features": {}
    }
    features = input_ops.build_feature_placeholders(config)
    self.assertFeatureShapesEqual(expected_shapes, features)

    # Two time series features.
    config = configdict.ConfigDict({
        "time_feature_1": {
            "length": 14,
            "is_time_series": True,
        },
        "time_feature_2": {
            "length": 5,
            "is_time_series": True,
        }
    })
    expected_shapes = {
        "time_series_features": {
            "time_feature_1": [None, 14],
            "time_feature_2": [None, 5],
        },
        "aux_features": {}
    }
    features = input_ops.build_feature_placeholders(config)
    self.assertFeatureShapesEqual(expected_shapes, features)

    # One aux feature.
    config = configdict.ConfigDict({
        "time_feature_1": {
            "length": 14,
            "is_time_series": True,
        },
        "aux_feature_1": {
            "length": 1,
            "is_time_series": False,
        }
    })
    expected_shapes = {
        "time_series_features": {
            "time_feature_1": [None, 14],
        },
        "aux_features": {
            "aux_feature_1": [None, 1]
        }
    }
    features = input_ops.build_feature_placeholders(config)
    self.assertFeatureShapesEqual(expected_shapes, features)

    # Two aux features.
    config = configdict.ConfigDict({
        "time_feature_1": {
            "length": 14,
            "is_time_series": True,
        },
        "aux_feature_1": {
            "length": 1,
            "is_time_series": False,
        },
        "aux_feature_2": {
            "length": 6,
            "is_time_series": False,
        },
    })
    expected_shapes = {
        "time_series_features": {
            "time_feature_1": [None, 14],
        },
        "aux_features": {
            "aux_feature_1": [None, 1],
            "aux_feature_2": [None, 6]
        }
    }
    features = input_ops.build_feature_placeholders(config)
    self.assertFeatureShapesEqual(expected_shapes, features)


if __name__ == "__main__":
  tf.test.main()

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""SSDFeatureExtractor for MobileNetEdgeTPU features."""

import tensorflow as tf

from object_detection.models import faster_rcnn_mobilenet_v3_feature_extractor
from nets.mobilenet import mobilenet_v3

slim = tf.contrib.slim


class FasterRCNNMobileNetEdgeTPUFeatureExtractor(
    faster_rcnn_mobilenet_v3_feature_extractor.FasterRCNNMobilenetV3FeatureExtractor):
  """MobileNetEdgeTPU feature extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               reuse_weights=None,
               scope_name='MobilenetEdgeTPU'):
    super(SSDMobileNetEdgeTPUFeatureExtractor, self).__init__(
        conv_defs=mobilenet_v3.V3_EDGETPU,
        from_layer=['layer_18/expansion_output', 'layer_23'],
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        reuse_weights=reuse_weights,
        scope_name=scope_name
    )

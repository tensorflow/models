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

# Lint as: python3
"""Tests for Yolo models."""

from official.vision.beta.projects.yolo.modeling.backbones.darknet import \
    Darknet
from official.vision.beta.projects.yolo.modeling.decoders.yolo_decoder import \
    YoloDecoder
from official.vision.beta.projects.yolo.modeling.heads.yolo_head import YoloHead
from official.vision.beta.projects.yolo.modeling import yolo_model

class YoloTest(parameterized.TestCase, tf.test.TestCase):
  def test_network_creation(self):
    backbone = ['darknettiny', 'darknet53', 'cspdarknet53', 'altered_cspdarknet53', 'cspdarknettiny', 'csp-large']
    base_model = yolo_model.YOLO_MODELS[version][decoder]

    backbone = Darknet(model_id=backbone)
    decoder = YoloDecoder(backbone.output_specs, **base_model)
    head = YoloHead(
        min_level=model_config.min_level,
        max_level=model_config.max_level,
        classes=model_config.num_classes,
        boxes_per_level=model_config.boxes_per_scale,
        norm_momentum=model_config.norm_activation.norm_momentum,
        norm_epsilon=model_config.norm_activation.norm_epsilon,
        kernel_regularizer=l2_regularization,
        smart_bias=model_config.smart_bias)
    pass

if __name__ == '__main__':
  tf.test.main()

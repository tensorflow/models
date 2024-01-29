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

"""Test for centernet detection model."""

from absl.testing import parameterized
import tensorflow as tf

from official.projects.centernet.configs import backbones
from official.projects.centernet.modeling import centernet_model
from official.projects.centernet.modeling.backbones import hourglass
from official.projects.centernet.modeling.heads import centernet_head
from official.projects.centernet.modeling.layers import detection_generator
from official.vision.configs import common


class CenterNetTest(parameterized.TestCase, tf.test.TestCase):

  def testBuildCenterNet(self):
    backbone = hourglass.build_hourglass(
        input_specs=tf.keras.layers.InputSpec(shape=[None, 512, 512, 3]),
        backbone_config=backbones.Backbone(type='hourglass'),
        norm_activation_config=common.NormActivation(use_sync_bn=True)
    )

    task_config = {
        'ct_heatmaps': 90,
        'ct_offset': 2,
        'ct_size': 2,
    }

    input_levels = ['2_0', '2']

    head = centernet_head.CenterNetHead(
        task_outputs=task_config,
        input_specs=backbone.output_specs,
        input_levels=input_levels)

    detection_ge = detection_generator.CenterNetDetectionGenerator()

    model = centernet_model.CenterNetModel(
        backbone=backbone,
        head=head,
        detection_generator=detection_ge
    )

    outputs = model(tf.zeros((5, 512, 512, 3)))
    self.assertLen(outputs['raw_output'], 3)
    self.assertLen(outputs['raw_output']['ct_heatmaps'], 2)
    self.assertLen(outputs['raw_output']['ct_offset'], 2)
    self.assertLen(outputs['raw_output']['ct_size'], 2)
    self.assertEqual(outputs['raw_output']['ct_heatmaps'][0].shape,
                     (5, 128, 128, 90))
    self.assertEqual(outputs['raw_output']['ct_offset'][0].shape,
                     (5, 128, 128, 2))
    self.assertEqual(outputs['raw_output']['ct_size'][0].shape,
                     (5, 128, 128, 2))


if __name__ == '__main__':
  tf.test.main()

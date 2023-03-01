# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for factory.py."""

import numpy as np
import tensorflow as tf

# pylint: disable=unused-import
from official.projects.yolo.configs import backbones
from official.projects.yolo.configs import yolo
from official.projects.yolo.modeling import factory
from official.projects.yolo.modeling.backbones import darknet
from official.projects.yolo.modeling.decoders import yolo_decoder
from official.projects.yolo.modeling.heads import yolo_head as heads
from official.projects.yolo.modeling.layers import detection_generator
# pylint: enable=unused-import


class FactoryTest(tf.test.TestCase):

  def test_yolo_builder(self):
    num_classes = 3
    input_size = 640
    input_specs = tf.keras.layers.InputSpec(
        shape=[None, input_size, input_size, 3])
    model_config = yolo.Yolo(
        num_classes=num_classes,
        head=yolo.YoloHead(smart_bias=True),
        anchor_boxes=yolo.AnchorBoxes(
            anchors_per_scale=3,
            boxes=[
                yolo.Box(box=[12, 16]),
                yolo.Box(box=[19, 36]),
                yolo.Box(box=[40, 28]),
                yolo.Box(box=[36, 75]),
                yolo.Box(box=[76, 55]),
                yolo.Box(box=[72, 146]),
                yolo.Box(box=[142, 110]),
                yolo.Box(box=[192, 243]),
                yolo.Box(box=[459, 401])
            ]))
    l2_regularizer = tf.keras.regularizers.l2(5e-5)

    yolo_model, _ = factory.build_yolo(
        input_specs=input_specs,
        model_config=model_config,
        l2_regularization=l2_regularizer)

    # Do forward pass.
    inputs = np.random.rand(2, input_size, input_size, 3)
    _ = yolo_model(inputs)


if __name__ == '__main__':
  tf.test.main()

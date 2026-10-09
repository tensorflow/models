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

"""Tests for tensorflow_models.official.projects.detr.detr."""
import tensorflow as tf
from official.projects.rngdet_plus.modeling import rngdet
from official.vision.modeling.backbones import resnet
from official.vision.modeling.decoders import fpn


class DetrTest(tf.test.TestCase):

  def test_forward(self):
    num_queries = 10
    hidden_size = 128
    num_classes = 2
    image_size = 128
    input_size = [image_size,image_size,3]
    batch_size = 64

    backbone = resnet.ResNet(50, bn_trainable=False)
    backbone_endpoint_name = '5'
    history_specs = tf.keras.layers.InputSpec(
        shape=[None] + input_size[:2] + [3])
    backbone_history = resnet.ResNet(50,
                                     input_specs=history_specs,
                                     bn_trainable=False)
    segment_fpn = fpn.FPN(backbone.output_specs,
                           min_level=2,
                           max_level=5)
    keypoint_fpn = fpn.FPN(backbone.output_specs,
                           min_level=2,
                           max_level=5)

    transformer = rngdet.DETRTransformer(
        hidden_size= hidden_size,
        num_encoder_layers=6,
        num_decoder_layers=6)

    multi_scale = rngdet.MultiScale( 
        transformer, 
        dim=transformer._hidden_size, 
        nheads=transformer._num_heads, 
        fpn_dims= [2048, 1024, 512, 256], 
        output_size = 128  )

    model = rngdet.RNGDet(backbone,
                      backbone_history,
                      backbone_endpoint_name,
                      segment_fpn,
                      keypoint_fpn,
                      transformer,
                      multi_scale,
                      num_queries,
                      hidden_size,
                      num_classes  ) 

    test_input = tf.ones((batch_size, image_size, image_size, 3))
    test_history = tf.ones((batch_size, image_size, image_size, 1))
    outs = model(test_input, test_history, training=True)

    self.assertLen(outs, 3)  # intermediate decoded outputs.

    self.assertAllEqual(
        tf.shape(outs[0]['cls_outputs']), (batch_size, num_queries, num_classes))
    self.assertAllEqual(
        tf.shape(outs[0]['box_outputs']), (batch_size, num_queries, num_classes))
    self.assertAllEqual(
        tf.shape(outs[1]), (batch_size, hidden_size, hidden_size, 1))
    self.assertAllEqual(
        tf.shape(outs[2]), (batch_size, hidden_size, hidden_size, 1))

if __name__ == '__main__':
  tf.test.main()
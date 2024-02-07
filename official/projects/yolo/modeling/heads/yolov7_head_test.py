# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for yolov7 heads."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.yolo.modeling.backbones import yolov7 as backbone
from official.projects.yolo.modeling.decoders import yolov7 as decoder
from official.projects.yolo.modeling.heads import yolov7_head as head

_INPUT_SIZE = (224, 224)


class YoloV7DetectionHeadTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ('yolov7',),
  )
  def test_network_creation(self, model_id):
    """Tests declaration of YOLOv7 detection head."""
    tf_keras.backend.set_image_data_format('channels_last')

    backbone_network = backbone.YoloV7(model_id)
    decoder_network = decoder.YoloV7(backbone_network.output_specs, model_id)
    head_network = head.YoloV7DetectionHead()

    inputs = tf_keras.Input(shape=(*_INPUT_SIZE, 3), batch_size=1)
    outputs = head_network(decoder_network(backbone_network(inputs)))

    for level, level_output in outputs.items():
      scale = 2 ** int(level)
      input_size = (_INPUT_SIZE[0] // scale, _INPUT_SIZE[1] // scale)
      head_config = head_network.get_config()
      num_classes = head_config['num_classes']
      num_anchors = head_config['num_anchors']
      self.assertAllEqual(
          (1, *input_size, num_anchors, num_classes + 5),
          level_output.shape.as_list(),
      )

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        num_classes=3,
        min_level=3,
        max_level=5,
        num_anchors=3,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_initializer='zeros',
        bias_regularizer=None,
    )
    network = head.YoloV7DetectionHead(**kwargs)

    # Create another network object from the first object's config.
    new_network = head.YoloV7DetectionHead.from_config(network.get_config())

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()

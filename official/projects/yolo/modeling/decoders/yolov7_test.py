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

"""Tests for yolov7 decoder."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.projects.yolo.modeling.backbones import yolov7 as backbone
from official.projects.yolo.modeling.decoders import yolov7 as decoder

_INPUT_SIZE = (224, 224)


class YoloV7DecoderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ('yolov7',),
  )
  def test_network_creation(self, model_id):
    """Tests declaration of YOLOv7 decoder variants."""
    tf.keras.backend.set_image_data_format('channels_last')

    backbone_network = backbone.YoloV7(model_id)
    decoder_network = decoder.YoloV7(backbone_network.output_specs, model_id)
    self.assertEqual(decoder_network.get_config()['model_id'], model_id)

    inputs = tf.keras.Input(shape=(*_INPUT_SIZE, 3), batch_size=1)
    outputs = decoder_network(backbone_network(inputs))

    for level, level_output in outputs.items():
      scale = 2 ** int(level)
      input_size = (_INPUT_SIZE[0] // scale, _INPUT_SIZE[1] // scale)
      self.assertAllEqual((1, *input_size), level_output.shape.as_list()[:-1])

  @combinations.generate(
      combinations.combine(
          strategy=[
              strategy_combinations.cloud_tpu_strategy,
              strategy_combinations.one_device_strategy_gpu,
          ],
      )
  )
  def test_sync_bn_multiple_devices(self, strategy):
    """Test for sync bn on TPU and GPU devices."""
    inputs = np.random.rand(1, *_INPUT_SIZE, 3)

    tf.keras.backend.set_image_data_format('channels_last')

    with strategy.scope():
      backbone_network = backbone.YoloV7(model_id='yolov7', use_sync_bn=True)
      decoder_network = decoder.YoloV7(
          backbone_network.output_specs, model_id='yolov7', use_sync_bn=True)
      _ = decoder_network(backbone_network(inputs))

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        model_id='yolov7',
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        activation='swish',
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_initializer='zeros',
        bias_regularizer=None,
    )
    backbone_network = backbone.YoloV7(**kwargs)
    kwargs['input_specs'] = backbone_network.output_specs
    decoder_network = decoder.YoloV7(**kwargs)

    # Create another network object from the first object's config.
    new_network = decoder.YoloV7.from_config(decoder_network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(decoder_network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()

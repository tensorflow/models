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

"""Tests for video classification network."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.vision.modeling import backbones
from official.vision.modeling import video_classification_model


class VideoClassificationNetworkTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (50, 8, 112, 'relu', False),
      (50, 8, 112, 'swish', True),
  )
  def test_resnet3d_network_creation(self, model_id, temporal_size,
                                     spatial_size, activation,
                                     aggregate_endpoints):
    """Test for creation of a ResNet3D-50 classifier."""
    input_specs = tf_keras.layers.InputSpec(
        shape=[None, temporal_size, spatial_size, spatial_size, 3])
    temporal_strides = [1, 1, 1, 1]
    temporal_kernel_sizes = [(3, 3, 3), (3, 1, 3, 1), (3, 1, 3, 1, 3, 1),
                             (1, 3, 1)]

    tf_keras.backend.set_image_data_format('channels_last')

    backbone = backbones.ResNet3D(
        model_id=model_id,
        temporal_strides=temporal_strides,
        temporal_kernel_sizes=temporal_kernel_sizes,
        input_specs=input_specs,
        activation=activation)

    num_classes = 1000
    model = video_classification_model.VideoClassificationModel(
        backbone=backbone,
        num_classes=num_classes,
        input_specs={'image': input_specs},
        dropout_rate=0.2,
        aggregate_endpoints=aggregate_endpoints,
    )

    inputs = np.random.rand(2, temporal_size, spatial_size, spatial_size, 3)
    logits = model(inputs)
    self.assertAllEqual([2, num_classes], logits.numpy().shape)

  def test_serialize_deserialize(self):
    """Validate the classification network can be serialized and deserialized."""
    model_id = 50
    temporal_strides = [1, 1, 1, 1]
    temporal_kernel_sizes = [(3, 3, 3), (3, 1, 3, 1), (3, 1, 3, 1, 3, 1),
                             (1, 3, 1)]

    backbone = backbones.ResNet3D(
        model_id=model_id,
        temporal_strides=temporal_strides,
        temporal_kernel_sizes=temporal_kernel_sizes)

    model = video_classification_model.VideoClassificationModel(
        backbone=backbone, num_classes=1000)

    config = model.get_config()
    new_model = video_classification_model.VideoClassificationModel.from_config(
        config)

    # Validate that the config can be forced to JSON.
    _ = new_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(model.get_config(), new_model.get_config())


if __name__ == '__main__':
  tf.test.main()

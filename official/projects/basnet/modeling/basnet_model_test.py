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

"""Tests for basnet network."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.projects.basnet.modeling import basnet_model
from official.projects.basnet.modeling import refunet


class BASNetNetworkTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (256),
      (512),
  )
  def test_basnet_network_creation(
      self, input_size):
    """Test for creation of a segmentation network."""
    inputs = np.random.rand(2, input_size, input_size, 3)
    tf.keras.backend.set_image_data_format('channels_last')

    backbone = basnet_model.BASNetEncoder()
    decoder = basnet_model.BASNetDecoder()
    refinement = refunet.RefUnet()

    model = basnet_model.BASNetModel(
        backbone=backbone,
        decoder=decoder,
        refinement=refinement
    )

    sigmoids = model(inputs)
    levels = sorted(sigmoids.keys())
    self.assertAllEqual(
        [2, input_size, input_size, 1],
        sigmoids[levels[-1]].numpy().shape)

  def test_serialize_deserialize(self):
    """Validate the network can be serialized and deserialized."""
    backbone = basnet_model.BASNetEncoder()
    decoder = basnet_model.BASNetDecoder()
    refinement = refunet.RefUnet()

    model = basnet_model.BASNetModel(
        backbone=backbone,
        decoder=decoder,
        refinement=refinement
    )

    config = model.get_config()
    new_model = basnet_model.BASNetModel.from_config(config)

    # Validate that the config can be forced to JSON.
    _ = new_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(model.get_config(), new_model.get_config())


if __name__ == '__main__':
  tf.test.main()

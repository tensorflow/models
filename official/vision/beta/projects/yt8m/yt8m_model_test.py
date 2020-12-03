# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for yt8m network."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.vision.beta.projects.yt8m import yt8m_model
from official.vision.beta.projects.yt8m.configs import yt8m as yt8m_cfg

class YT8MNetworkTest(parameterized.TestCase, tf.test.TestCase):

  # test_yt8m_network_creation arbitrary params
  @parameterized.parameters(
      (1,1,1),
      (1,1,1),
  )
  def test_yt8m_network_creation(self, num_frames, height, width):
    """Test for creation of a YT8M Model."""

    # None part : batch * num_test_clips
    input_specs = tf.keras.layers.InputSpec(
        shape=[None, num_frames, height, width, 3]) 

    tf.keras.backend.set_image_data_format('channels_last')

    num_classes = 3862
    model = yt8m_model.YT8MModel(
      input_params=yt8m_cfg.YT8MModel,
      input_specs=input_specs,
      num_frames=num_frames,
      num_classes=num_classes,
    )

    # batch * num_test_clips = 2 -> arbitrary value for test
    inputs = np.random.rand(2, num_frames, height, width, 3)
    logits = model(inputs)
    self.assertAllEqual([2, num_classes], logits.numpy().shape) # expected, actual

  def test_serialize_deserialize(self):
    """Validate the classification network can be serialized and deserialized."""

    model = yt8m_model.YT8MModel(
      input_params=yt8m_cfg.YT8MModel,
      num_classes=3862
    )

    config = model.get_config()
    new_model = yt8m_model.YT8MModel.from_config(config)

    # Validate that the config can be forced to JSON.
    _ = new_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(model.get_config(), new_model.get_config())


if __name__ == '__main__':
  tf.test.main()

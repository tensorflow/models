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

"""Tests for yt8m network."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.projects.yt8m.configs import yt8m as yt8m_cfg
from official.projects.yt8m.modeling import yt8m_model


class YT8MNetworkTest(parameterized.TestCase, tf.test.TestCase):
  """Class for testing yt8m network."""

  # test_yt8m_network_creation arbitrary params
  @parameterized.parameters((32, 1152), (24, 1152))  # 1152 = 1024 + 128
  def test_yt8m_network_creation(self, num_frames, feature_dims):
    """Test for creation of a YT8M Model.

    Args:
      num_frames: number of frames.
      feature_dims: indicates total dimension size of the features.
    """
    input_specs = tf.keras.layers.InputSpec(shape=[None, None, feature_dims])

    num_classes = 3862
    model = yt8m_model.VideoClassificationModel(
        params=yt8m_cfg.YT8MTask().model,
        num_classes=num_classes,
        input_specs=input_specs,
    )

    # batch = 2 -> arbitrary value for test.
    inputs = np.random.rand(2, num_frames, feature_dims)
    predictions = model(inputs)['predictions']
    self.assertAllEqual([2, num_classes], predictions.numpy().shape)

  def test_serialize_deserialize(self):
    model = yt8m_model.VideoClassificationModel(
        params=yt8m_cfg.YT8MTask().model
    )

    config = model.get_config()
    new_model = yt8m_model.VideoClassificationModel.from_config(config)

    # If the serialization was successful,
    # the new config should match the old.
    self.assertAllEqual(model.get_config(), new_model.get_config())


if __name__ == '__main__':
  tf.test.main()

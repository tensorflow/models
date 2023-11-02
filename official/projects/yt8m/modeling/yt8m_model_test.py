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
import tensorflow as tf, tf_keras

from official.projects.yt8m.configs import yt8m as yt8m_cfg
from official.projects.yt8m.modeling import yt8m_model


class YT8MNetworkTest(parameterized.TestCase, tf.test.TestCase):
  """Class for testing yt8m network."""

  # test_yt8m_network_creation arbitrary params
  @parameterized.product(
      num_sample_frames=(None, 16, 32),
      pooling_method=('average', 'max', 'swap'),
  )
  def test_yt8m_network_creation(
      self, num_sample_frames, pooling_method
  ):
    """Test for creation of a YT8M Model.

    Args:
      num_sample_frames: indicates number of frames to sample.
      pooling_method: str of frame pooling method.
    """
    num_frames = 24
    feature_dims = 52
    num_classes = 45
    input_specs = tf_keras.layers.InputSpec(shape=[None, None, feature_dims])

    params = yt8m_cfg.YT8MTask().model
    params.backbone.dbof.pooling_method = pooling_method
    model = yt8m_model.VideoClassificationModel(
        params=params,
        num_classes=num_classes,
        input_specs=input_specs,
    )

    # batch = 2 -> arbitrary value for test.
    if num_sample_frames:
      inputs = np.random.rand(2, num_sample_frames, feature_dims)
      num_frames = tf.constant([num_sample_frames, num_sample_frames])
    else:
      # Add padding frames.
      inputs = np.random.rand(2, num_frames + 4, feature_dims)
      num_frames = tf.constant([num_frames, num_frames + 1])

    predictions = model(inputs, num_frames=num_frames)['predictions']
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

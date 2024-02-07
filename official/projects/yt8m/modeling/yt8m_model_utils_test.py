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

"""Tests for YT8M modeling utilities."""
from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.yt8m.modeling import yt8m_model_utils


class Yt8MModelUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      frame_pooling=("average", "max", "swap", "none"),
      use_frame_mask=(True, False),
  )
  def test_frame_pooling(self, frame_pooling, use_frame_mask):
    frame = tf.constant([
        [[0.0, 0.0, 0.0], [0.0, 1.0, -1.0]],
        [[0.0, 0.0, 0.0], [0.0, 2.0, -2.0]],
    ])
    num_frames = tf.constant([2, 2]) if use_frame_mask else None
    pooled_frame = yt8m_model_utils.frame_pooling(
        frame, method=frame_pooling, num_frames=num_frames
    )
    if frame_pooling == "swap":
      self.assertAllClose([[0.0, 1.0, -1.0], [0.0, 2.0, -2.0]], pooled_frame)
    elif frame_pooling == "average":
      self.assertAllClose([[0.0, 0.5, -0.5], [0.0, 1.0, -1.0]], pooled_frame)
    elif frame_pooling == "max":
      self.assertAllClose([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], pooled_frame)
    elif frame_pooling == "none":
      self.assertAllClose(
          [
              [0.0, 0.0, 0.0],
              [0.0, 1.0, -1.0],
              [0.0, 0.0, 0.0],
              [0.0, 2.0, -2.0],
          ],
          pooled_frame,
      )


if __name__ == "__main__":
  tf.test.main()

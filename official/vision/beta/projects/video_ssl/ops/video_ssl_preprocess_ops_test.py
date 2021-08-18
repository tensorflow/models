# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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


import tensorflow as tf
from official.vision.beta.ops import preprocess_ops_3d
from official.vision.beta.projects.video_ssl.ops import video_ssl_preprocess_ops


class VideoSslPreprocessOpsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._raw_frames = tf.random.uniform((250, 256, 256, 3), minval=0,
                                         maxval=255, dtype=tf.dtypes.int32)
    self._sampled_frames = self._raw_frames[:16]
    self._frames = preprocess_ops_3d.normalize_image(
        self._sampled_frames, False, tf.float32)

  def test_sample_ssl_sequence(self):
    sampled_seq = video_ssl_preprocess_ops.sample_ssl_sequence(
        self._raw_frames, 16, True, 2)
    self.assertAllEqual(sampled_seq.shape, (32, 256, 256, 3))

  def test_random_color_jitter_3d(self):
    jittered_clip = video_ssl_preprocess_ops.random_color_jitter_3d(
        self._frames)
    self.assertAllEqual(jittered_clip.shape, (16, 256, 256, 3))

  def test_random_blur_3d(self):
    blurred_clip = video_ssl_preprocess_ops.random_blur_3d(
        self._frames, 256, 256)
    self.assertAllEqual(blurred_clip.shape, (16, 256, 256, 3))

if __name__ == '__main__':
  tf.test.main()

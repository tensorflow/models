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

"""kmeans_test tests."""
from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.projects.yolo.ops import kmeans_anchors


class KMeansTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((9, 3, 100))
  def test_kmeans(self, k, anchors_per_scale, samples):
    sample_list = []
    for _ in range(samples):
      boxes = tf.convert_to_tensor(np.random.uniform(0, 1, [k * 100, 4]))
      sample_list.append({
          "groundtruth_boxes": boxes,
          "width": 10,
          "height": 10
      })

    kmeans = kmeans_anchors.AnchorKMeans()
    cl = kmeans(
        sample_list, k, anchors_per_scale, image_resolution=[512, 512, 3])
    cl = tf.convert_to_tensor(cl)
    self.assertAllEqual(tf.shape(cl).numpy(), [k, 2])


if __name__ == "__main__":
  tf.test.main()

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

import tensorflow as tf, tf_keras
from official.projects.waste_identification_ml.model_inference import preprocessing


class PreprocessingTest(tf.test.TestCase):

  def test_normalize_image(self):
    image = tf.convert_to_tensor(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.int8
    )
    expected = tf.convert_to_tensor(
        value=[
            [
                [-2.0835197, -1.9654106, -1.6994576],
                [-1.9803666, -1.859955, -1.5944706],
            ],
            [
                [-1.8772135, -1.7544994, -1.4894838],
                [-1.7740605, -1.6490439, -1.384497],
            ],
        ],
        dtype=tf.float32,
    )

    result = preprocessing.normalize_image(image=image)

    self.assertAllEqual(expected, result)

  def test_normalize_scaled_float_image(self):
    image = tf.convert_to_tensor(
        [
            [[0.00787, 0.01575, 0.02362], [0.0315, 0.03937, 0.04724]],
            [[0.0551, 0.063, 0.07086], [0.07874, 0.0866, 0.0945]],
        ],
        dtype=tf.float32,
    )
    expected = tf.convert_to_tensor(
        value=[
            [
                [-2.0835197, -1.9654106, -1.6994576],
                [-1.9803666, -1.859955, -1.5944706],
            ],
            [
                [-1.8772135, -1.7544994, -1.4894838],
                [-1.7740605, -1.6490439, -1.384497],
            ],
        ],
        dtype=tf.float32,
    )

    result = preprocessing.normalize_scaled_float_image(image=image)

    self.assertAllCloseAccordingToType(expected, result, rtol=1e-4)


if __name__ == "__main__":
  tf.test.main()

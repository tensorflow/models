# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

import unittest
import numpy as np
from official.projects.waste_identification_ml.Deploy.detr_cloud_deployment.client import color_extraction


class ColorExtractionTest(unittest.TestCase):

  def test_find_dominant_color_with_non_black_pixels(self):
    # Create an image with a clear dominant color (Red)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[0:5, 0:5] = [255, 0, 0]  # Top-left quarter is Red
    image[5:10, 5:10] = [100, 0, 0]  # Bottom-right quarter is dark Red
    dominant_color = color_extraction.find_dominant_color(image)
    self.assertEqual(dominant_color, (177, 0, 0))

  def test_find_dominant_color_with_only_black_pixels(self):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    dominant_color = color_extraction.find_dominant_color(
        image, black_threshold=50
    )
    self.assertEqual(dominant_color, (0, 0, 0))

  def test_rgb_int_to_lab(self):
    rgb = (255, 255, 255)
    lab = color_extraction.rgb_int_to_lab(rgb)
    # White in LAB is approx (100, 0, 0)
    self.assertIsInstance(lab, np.ndarray)
    self.assertEqual(lab.shape, (3,))
    np.testing.assert_allclose(lab, [100.0, 0.0, 0.0], atol=1e-2)

  def test_color_distance(self):
    color_a = (100, 0, 0)  # LAB
    color_b = (100, 0, 0)  # LAB
    distance = color_extraction.color_distance(color_a, color_b)
    self.assertEqual(distance, 0.0)
    color_c = (0, 0, 0)
    distance_diff = color_extraction.color_distance(color_a, color_c)
    self.assertGreater(distance_diff, 0.0)

  def test_build_color_lab_list(self):
    generic_colors = [('black', '#000000'), ('white', '#ffffff')]
    names, lab_values = color_extraction.build_color_lab_list(generic_colors)
    np.testing.assert_array_equal(names, ['black', 'white'])
    self.assertEqual(len(lab_values), 2)
    np.testing.assert_allclose(lab_values[0], [0.0, 0.0, 0.0], atol=1e-2)
    np.testing.assert_allclose(lab_values[1], [100.0, 0.0, 0.0], atol=1e-2)

  def test_get_generic_color_name(self):
    rgb_colors = [(255, 0, 0), (0, 0, 255)]  # Red, Blue
    generic_colors = [
        ('red', '#ff0000'),
        ('blue', '#0000ff'),
        ('green', '#00ff00'),
    ]
    names = color_extraction.get_generic_color_name(rgb_colors, generic_colors)
    self.assertEqual(names, ['red', 'blue'])


if __name__ == '__main__':
  unittest.main()

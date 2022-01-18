# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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


"""Tests for mask_ops.py."""

# Import libraries
import numpy as np
import tensorflow as tf
from official.vision.beta.ops import mask_ops


class MaskUtilsTest(tf.test.TestCase):

  def testPasteInstanceMasks(self):
    image_height = 10
    image_width = 10
    mask_height = 6
    mask_width = 6
    masks = np.random.randint(0, 255, (1, mask_height, mask_width))
    detected_boxes = np.array([[0.0, 2.0, mask_width, mask_height]])

    _ = mask_ops.paste_instance_masks(
        masks, detected_boxes, image_height, image_width)

  def testPasteInstanceMasksV2(self):
    image_height = 10
    image_width = 10
    mask_height = 6
    mask_width = 6
    masks = np.random.randint(0, 255, (1, mask_height, mask_width))
    detected_boxes = np.array([[0.0, 2.0, mask_width, mask_height]])

    image_masks = mask_ops.paste_instance_masks_v2(
        masks, detected_boxes, image_height, image_width)

    self.assertNDArrayNear(
        image_masks[:, 2:8, 0:6],
        np.array(masks > 0.5, dtype=np.uint8),
        1e-5)


if __name__ == '__main__':
  tf.test.main()

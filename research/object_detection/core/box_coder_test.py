# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for object_detection.core.box_coder."""

import tensorflow as tf

from object_detection.core import box_coder
from object_detection.core import box_list


class MockBoxCoder(box_coder.BoxCoder):
  """Test BoxCoder that encodes/decodes using the multiply-by-two function."""

  def code_size(self):
    return 4

  def _encode(self, boxes, anchors):
    return 2.0 * boxes.get()

  def _decode(self, rel_codes, anchors):
    return box_list.BoxList(rel_codes / 2.0)


class BoxCoderTest(tf.test.TestCase):

  def test_batch_decode(self):
    mock_anchor_corners = tf.constant(
        [[0, 0.1, 0.2, 0.3], [0.2, 0.4, 0.4, 0.6]], tf.float32)
    mock_anchors = box_list.BoxList(mock_anchor_corners)
    mock_box_coder = MockBoxCoder()

    expected_boxes = [[[0.0, 0.1, 0.5, 0.6], [0.5, 0.6, 0.7, 0.8]],
                      [[0.1, 0.2, 0.3, 0.4], [0.7, 0.8, 0.9, 1.0]]]

    encoded_boxes_list = [mock_box_coder.encode(
        box_list.BoxList(tf.constant(boxes)), mock_anchors)
                          for boxes in expected_boxes]
    encoded_boxes = tf.stack(encoded_boxes_list)
    decoded_boxes = box_coder.batch_decode(
        encoded_boxes, mock_box_coder, mock_anchors)

    with self.test_session() as sess:
      decoded_boxes_result = sess.run(decoded_boxes)
      self.assertAllClose(expected_boxes, decoded_boxes_result)


if __name__ == '__main__':
  tf.test.main()

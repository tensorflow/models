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

"""Mean stddev box coder.

This box coder use the following coding schema to encode boxes:
rel_code = (box_corner - anchor_corner_mean) / anchor_corner_stddev.
"""
from object_detection.core import box_coder
from object_detection.core import box_list


class MeanStddevBoxCoder(box_coder.BoxCoder):
  """Mean stddev box coder."""

  def __init__(self, stddev=0.01):
    """Constructor for MeanStddevBoxCoder.

    Args:
      stddev: The standard deviation used to encode and decode boxes.
    """
    self._stddev = stddev

  @property
  def code_size(self):
    return 4

  def _encode(self, boxes, anchors):
    """Encode a box collection with respect to anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of N anchors.

    Returns:
      a tensor representing N anchor-encoded boxes

    Raises:
      ValueError: if the anchors still have deprecated stddev field.
    """
    box_corners = boxes.get()
    if anchors.has_field('stddev'):
      raise ValueError("'stddev' is a parameter of MeanStddevBoxCoder and "
                       "should not be specified in the box list.")
    means = anchors.get()
    return (box_corners - means) / self._stddev

  def _decode(self, rel_codes, anchors):
    """Decode.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes

    Raises:
      ValueError: if the anchors still have deprecated stddev field and expects
        the decode method to use stddev value from that field.
    """
    means = anchors.get()
    if anchors.has_field('stddev'):
      raise ValueError("'stddev' is a parameter of MeanStddevBoxCoder and "
                       "should not be specified in the box list.")
    box_corners = rel_codes * self._stddev + means
    return box_list.BoxList(box_corners)

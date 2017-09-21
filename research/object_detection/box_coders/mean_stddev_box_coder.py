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

  @property
  def code_size(self):
    return 4

  def _encode(self, boxes, anchors):
    """Encode a box collection with respect to anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of N anchors.  We assume that anchors has an associated
        stddev field.

    Returns:
      a tensor representing N anchor-encoded boxes
    Raises:
      ValueError: if the anchors BoxList does not have a stddev field
    """
    if not anchors.has_field('stddev'):
      raise ValueError('anchors must have a stddev field')
    box_corners = boxes.get()
    means = anchors.get()
    stddev = anchors.get_field('stddev')
    return (box_corners - means) / stddev

  def _decode(self, rel_codes, anchors):
    """Decode.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.  We assume that anchors has an associated
        stddev field.

    Returns:
      boxes: BoxList holding N bounding boxes
    Raises:
      ValueError: if the anchors BoxList does not have a stddev field
    """
    if not anchors.has_field('stddev'):
      raise ValueError('anchors must have a stddev field')
    means = anchors.get()
    stddevs = anchors.get_field('stddev')
    box_corners = rel_codes * stddevs + means
    return box_list.BoxList(box_corners)

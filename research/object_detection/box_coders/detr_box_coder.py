# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""DETR box coder.

DETR box coder follows the coding schema described below:
  ty = y
  tx = x
  th = h
  tw = w
  where x, y, w, h denote the box's center coordinates, width and height
  respectively. Similarly, xa, ya, wa, ha denote the anchor's center
  coordinates, width and height. tx, ty, tw and th denote the anchor-encoded
  center, width and height respectively.

  See http://arxiv.org/abs/1506.01497 for details.
"""

import tensorflow.compat.v1 as tf

from object_detection.core import box_coder
from object_detection.core import box_list

EPSILON = 1e-8


class DETRBoxCoder(box_coder.BoxCoder):
  """Faster RCNN box coder."""

  def __init__(self):
    """Constructor for DETRBoxCoder.

    Args:
      scale_factors: List of 4 positive scalars to scale ty, tx, th and tw.
        If set to None, does not perform scaling. For Faster RCNN,
        the open-source implementation recommends using [10.0, 10.0, 5.0, 5.0].
    """
    pass

  @property
  def code_size(self):
    return 4

  def _encode(self, boxes, anchors):
    """Encode a box collection with respect to anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of anchors.

    Returns:
      a tensor representing N anchor-encoded boxes of the format
      [ty, tx, th, tw].
    """
    # Convert anchors to the center coordinate representation.
    ty, tx, th, tw = boxes.get_center_coordinates_and_sizes()
    return tf.transpose(tf.stack([ty, tx, th, tw]))

  def _decode(self, rel_codes, anchors):
    """Decode relative codes to boxes.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes.
    """
    ty, tx, th, tw = tf.unstack(tf.transpose(rel_codes))
    ymin = ty - th / 2.
    xmin = tx - tw / 2.
    ymax = ty + th / 2.
    xmax = tx + tw / 2.
    return box_list.BoxList(tf.transpose(tf.stack([ymin, xmin, ymax, xmax])))


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

"""Square box coder.

Square box coder follows the coding schema described below:
l = sqrt(h * w)
la = sqrt(ha * wa)
ty = (y - ya) / la
tx = (x - xa) / la
tl = log(l / la)
where x, y, w, h denote the box's center coordinates, width, and height,
respectively. Similarly, xa, ya, wa, ha denote the anchor's center
coordinates, width and height. tx, ty, tl denote the anchor-encoded
center, and length, respectively. Because the encoded box is a square, only
one length is encoded.

This has shown to provide performance improvements over the Faster RCNN box
coder when the objects being detected tend to be square (e.g. faces) and when
the input images are not distorted via resizing.
"""

import tensorflow as tf

from object_detection.core import box_coder
from object_detection.core import box_list

EPSILON = 1e-8


class SquareBoxCoder(box_coder.BoxCoder):
  """Encodes a 3-scalar representation of a square box."""

  def __init__(self, scale_factors=None):
    """Constructor for SquareBoxCoder.

    Args:
      scale_factors: List of 3 positive scalars to scale ty, tx, and tl.
        If set to None, does not perform scaling. For faster RCNN,
        the open-source implementation recommends using [10.0, 10.0, 5.0].

    Raises:
      ValueError: If scale_factors is not length 3 or contains values less than
        or equal to 0.
    """
    if scale_factors:
      if len(scale_factors) != 3:
        raise ValueError('The argument scale_factors must be a list of length '
                         '3.')
      if any(scalar <= 0 for scalar in scale_factors):
        raise ValueError('The values in scale_factors must all be greater '
                         'than 0.')
    self._scale_factors = scale_factors

  @property
  def code_size(self):
    return 3

  def _encode(self, boxes, anchors):
    """Encodes a box collection with respect to an anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of anchors.

    Returns:
      a tensor representing N anchor-encoded boxes of the format
      [ty, tx, tl].
    """
    # Convert anchors to the center coordinate representation.
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    la = tf.sqrt(ha * wa)
    ycenter, xcenter, h, w = boxes.get_center_coordinates_and_sizes()
    l = tf.sqrt(h * w)
    # Avoid NaN in division and log below.
    la += EPSILON
    l += EPSILON

    tx = (xcenter - xcenter_a) / la
    ty = (ycenter - ycenter_a) / la
    tl = tf.log(l / la)
    # Scales location targets for joint training.
    if self._scale_factors:
      ty *= self._scale_factors[0]
      tx *= self._scale_factors[1]
      tl *= self._scale_factors[2]
    return tf.transpose(tf.stack([ty, tx, tl]))

  def _decode(self, rel_codes, anchors):
    """Decodes relative codes to boxes.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes.
    """
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    la = tf.sqrt(ha * wa)

    ty, tx, tl = tf.unstack(tf.transpose(rel_codes))
    if self._scale_factors:
      ty /= self._scale_factors[0]
      tx /= self._scale_factors[1]
      tl /= self._scale_factors[2]
    l = tf.exp(tl) * la
    ycenter = ty * la + ycenter_a
    xcenter = tx * la + xcenter_a
    ymin = ycenter - l / 2.
    xmin = xcenter - l / 2.
    ymax = ycenter + l / 2.
    xmax = xcenter + l / 2.
    return box_list.BoxList(tf.transpose(tf.stack([ymin, xmin, ymax, xmax])))

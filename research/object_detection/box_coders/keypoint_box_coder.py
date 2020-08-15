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

"""Keypoint box coder.

The keypoint box coder follows the coding schema described below (this is
similar to the FasterRcnnBoxCoder, except that it encodes keypoints in addition
to box coordinates):
  ty = (y - ya) / ha
  tx = (x - xa) / wa
  th = log(h / ha)
  tw = log(w / wa)
  tky0 = (ky0 - ya) / ha
  tkx0 = (kx0 - xa) / wa
  tky1 = (ky1 - ya) / ha
  tkx1 = (kx1 - xa) / wa
  ...
  where x, y, w, h denote the box's center coordinates, width and height
  respectively. Similarly, xa, ya, wa, ha denote the anchor's center
  coordinates, width and height. tx, ty, tw and th denote the anchor-encoded
  center, width and height respectively. ky0, kx0, ky1, kx1, ... denote the
  keypoints' coordinates, and tky0, tkx0, tky1, tkx1, ... denote the
  anchor-encoded keypoint coordinates.
"""

import tensorflow.compat.v1 as tf

from object_detection.core import box_coder
from object_detection.core import box_list
from object_detection.core import standard_fields as fields

EPSILON = 1e-8


class KeypointBoxCoder(box_coder.BoxCoder):
  """Keypoint box coder."""

  def __init__(self, num_keypoints, scale_factors=None):
    """Constructor for KeypointBoxCoder.

    Args:
      num_keypoints: Number of keypoints to encode/decode.
      scale_factors: List of 4 positive scalars to scale ty, tx, th and tw.
        In addition to scaling ty and tx, the first 2 scalars are used to scale
        the y and x coordinates of the keypoints as well. If set to None, does
        not perform scaling.
    """
    self._num_keypoints = num_keypoints

    if scale_factors:
      assert len(scale_factors) == 4
      for scalar in scale_factors:
        assert scalar > 0
    self._scale_factors = scale_factors
    self._keypoint_scale_factors = None
    if scale_factors is not None:
      self._keypoint_scale_factors = tf.expand_dims(
          tf.tile([
              tf.cast(scale_factors[0], dtype=tf.float32),
              tf.cast(scale_factors[1], dtype=tf.float32)
          ], [num_keypoints]), 1)

  @property
  def code_size(self):
    return 4 + self._num_keypoints * 2

  def _encode(self, boxes, anchors):
    """Encode a box and keypoint collection with respect to anchor collection.

    Args:
      boxes: BoxList holding N boxes and keypoints to be encoded. Boxes are
        tensors with the shape [N, 4], and keypoints are tensors with the shape
        [N, num_keypoints, 2].
      anchors: BoxList of anchors.

    Returns:
      a tensor representing N anchor-encoded boxes of the format
      [ty, tx, th, tw, tky0, tkx0, tky1, tkx1, ...] where tky0 and tkx0
      represent the y and x coordinates of the first keypoint, tky1 and tkx1
      represent the y and x coordinates of the second keypoint, and so on.
    """
    # Convert anchors to the center coordinate representation.
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    ycenter, xcenter, h, w = boxes.get_center_coordinates_and_sizes()
    keypoints = boxes.get_field(fields.BoxListFields.keypoints)
    keypoints = tf.transpose(tf.reshape(keypoints,
                                        [-1, self._num_keypoints * 2]))
    num_boxes = boxes.num_boxes()

    # Avoid NaN in division and log below.
    ha += EPSILON
    wa += EPSILON
    h += EPSILON
    w += EPSILON

    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha
    tw = tf.log(w / wa)
    th = tf.log(h / ha)

    tiled_anchor_centers = tf.tile(
        tf.stack([ycenter_a, xcenter_a]), [self._num_keypoints, 1])
    tiled_anchor_sizes = tf.tile(
        tf.stack([ha, wa]), [self._num_keypoints, 1])
    tkeypoints = (keypoints - tiled_anchor_centers) / tiled_anchor_sizes

    # Scales location targets as used in paper for joint training.
    if self._scale_factors:
      ty *= self._scale_factors[0]
      tx *= self._scale_factors[1]
      th *= self._scale_factors[2]
      tw *= self._scale_factors[3]
      tkeypoints *= tf.tile(self._keypoint_scale_factors, [1, num_boxes])

    tboxes = tf.stack([ty, tx, th, tw])
    return tf.transpose(tf.concat([tboxes, tkeypoints], 0))

  def _decode(self, rel_codes, anchors):
    """Decode relative codes to boxes and keypoints.

    Args:
      rel_codes: a tensor with shape [N, 4 + 2 * num_keypoints] representing N
        anchor-encoded boxes and keypoints
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes and keypoints.
    """
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()

    num_codes = tf.shape(rel_codes)[0]
    result = tf.unstack(tf.transpose(rel_codes))
    ty, tx, th, tw = result[:4]
    tkeypoints = result[4:]
    if self._scale_factors:
      ty /= self._scale_factors[0]
      tx /= self._scale_factors[1]
      th /= self._scale_factors[2]
      tw /= self._scale_factors[3]
      tkeypoints /= tf.tile(self._keypoint_scale_factors, [1, num_codes])

    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    decoded_boxes_keypoints = box_list.BoxList(
        tf.transpose(tf.stack([ymin, xmin, ymax, xmax])))

    tiled_anchor_centers = tf.tile(
        tf.stack([ycenter_a, xcenter_a]), [self._num_keypoints, 1])
    tiled_anchor_sizes = tf.tile(
        tf.stack([ha, wa]), [self._num_keypoints, 1])
    keypoints = tkeypoints * tiled_anchor_sizes + tiled_anchor_centers
    keypoints = tf.reshape(tf.transpose(keypoints),
                           [-1, self._num_keypoints, 2])
    decoded_boxes_keypoints.add_field(fields.BoxListFields.keypoints, keypoints)
    return decoded_boxes_keypoints

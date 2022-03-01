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

"""Utility functions for segmentations."""

import math
# Import libraries
import cv2
import numpy as np


def paste_instance_masks(masks,
                         detected_boxes,
                         image_height,
                         image_width):
  """Paste instance masks to generate the image segmentation results.

  Args:
    masks: a numpy array of shape [N, mask_height, mask_width] representing the
      instance masks w.r.t. the `detected_boxes`.
    detected_boxes: a numpy array of shape [N, 4] representing the reference
      bounding boxes.
    image_height: an integer representing the height of the image.
    image_width: an integer representing the width of the image.

  Returns:
    segms: a numpy array of shape [N, image_height, image_width] representing
      the instance masks *pasted* on the image canvas.
  """

  def expand_boxes(boxes, scale):
    """Expands an array of boxes by a given scale."""
    # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L227  # pylint: disable=line-too-long
    # The `boxes` in the reference implementation is in [x1, y1, x2, y2] form,
    # whereas `boxes` here is in [x1, y1, w, h] form
    w_half = boxes[:, 2] * .5
    h_half = boxes[:, 3] * .5
    x_c = boxes[:, 0] + w_half
    y_c = boxes[:, 1] + h_half

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp

  # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/test.py#L812  # pylint: disable=line-too-long
  # To work around an issue with cv2.resize (it seems to automatically pad
  # with repeated border values), we manually zero-pad the masks by 1 pixel
  # prior to resizing back to the original image resolution. This prevents
  # "top hat" artifacts. We therefore need to expand the reference boxes by an
  # appropriate factor.
  _, mask_height, mask_width = masks.shape
  scale = max((mask_width + 2.0) / mask_width,
              (mask_height + 2.0) / mask_height)

  ref_boxes = expand_boxes(detected_boxes, scale)
  ref_boxes = ref_boxes.astype(np.int32)
  padded_mask = np.zeros((mask_height + 2, mask_width + 2), dtype=np.float32)
  segms = []
  for mask_ind, mask in enumerate(masks):
    im_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    # Process mask inside bounding boxes.
    padded_mask[1:-1, 1:-1] = mask[:, :]

    ref_box = ref_boxes[mask_ind, :]
    w = ref_box[2] - ref_box[0] + 1
    h = ref_box[3] - ref_box[1] + 1
    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    mask = cv2.resize(padded_mask, (w, h))
    mask = np.array(mask > 0.5, dtype=np.uint8)

    x_0 = min(max(ref_box[0], 0), image_width)
    x_1 = min(max(ref_box[2] + 1, 0), image_width)
    y_0 = min(max(ref_box[1], 0), image_height)
    y_1 = min(max(ref_box[3] + 1, 0), image_height)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - ref_box[1]):(y_1 - ref_box[1]),
        (x_0 - ref_box[0]):(x_1 - ref_box[0])
    ]
    segms.append(im_mask)

  segms = np.array(segms)
  assert masks.shape[0] == segms.shape[0]
  return segms


def paste_instance_masks_v2(masks,
                            detected_boxes,
                            image_height,
                            image_width):
  """Paste instance masks to generate the image segmentation (v2).

  Args:
    masks: a numpy array of shape [N, mask_height, mask_width] representing the
      instance masks w.r.t. the `detected_boxes`.
    detected_boxes: a numpy array of shape [N, 4] representing the reference
      bounding boxes.
    image_height: an integer representing the height of the image.
    image_width: an integer representing the width of the image.

  Returns:
    segms: a numpy array of shape [N, image_height, image_width] representing
      the instance masks *pasted* on the image canvas.
  """
  _, mask_height, mask_width = masks.shape

  segms = []
  for i, mask in enumerate(masks):
    box = detected_boxes[i, :]
    xmin = box[0]
    ymin = box[1]
    xmax = xmin + box[2]
    ymax = ymin + box[3]

    # Sample points of the cropped mask w.r.t. the image grid.
    # Note that these coordinates may fall beyond the image.
    # Pixel clipping will happen after warping.
    xmin_int = int(math.floor(xmin))
    xmax_int = int(math.ceil(xmax))
    ymin_int = int(math.floor(ymin))
    ymax_int = int(math.ceil(ymax))

    alpha = box[2] / (1.0 * mask_width)
    beta = box[3] / (1.0 * mask_height)
    # pylint: disable=invalid-name
    # Transformation from mask pixel indices to image coordinate.
    M_mask_to_image = np.array(
        [[alpha, 0, xmin],
         [0, beta, ymin],
         [0, 0, 1]],
        dtype=np.float32)
    # Transformation from image to cropped mask coordinate.
    M_image_to_crop = np.array(
        [[1, 0, -xmin_int],
         [0, 1, -ymin_int],
         [0, 0, 1]],
        dtype=np.float32)
    M = np.dot(M_image_to_crop, M_mask_to_image)
    # Compensate the half pixel offset that OpenCV has in the
    # warpPerspective implementation: the top-left pixel is sampled
    # at (0,0), but we want it to be at (0.5, 0.5).
    M = np.dot(
        np.dot(
            np.array([[1, 0, -0.5],
                      [0, 1, -0.5],
                      [0, 0, 1]], np.float32),
            M),
        np.array([[1, 0, 0.5],
                  [0, 1, 0.5],
                  [0, 0, 1]], np.float32))
    # pylint: enable=invalid-name
    cropped_mask = cv2.warpPerspective(
        mask.astype(np.float32), M,
        (xmax_int - xmin_int, ymax_int - ymin_int))
    cropped_mask = np.array(cropped_mask > 0.5, dtype=np.uint8)

    img_mask = np.zeros((image_height, image_width))
    x0 = max(min(xmin_int, image_width), 0)
    x1 = max(min(xmax_int, image_width), 0)
    y0 = max(min(ymin_int, image_height), 0)
    y1 = max(min(ymax_int, image_height), 0)
    img_mask[y0:y1, x0:x1] = cropped_mask[
        (y0 - ymin_int):(y1 - ymin_int),
        (x0 - xmin_int):(x1 - xmin_int)]

    segms.append(img_mask)

  segms = np.array(segms)
  return segms


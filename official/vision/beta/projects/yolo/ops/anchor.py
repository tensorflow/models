# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Yolo Anchor labler."""
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import maximum, minimum
from official.vision.beta.projects.yolo.ops import box_ops
from official.vision.beta.projects.yolo.ops import preprocessing_ops
from official.vision.beta.projects.yolo.ops import loss_utils


def get_best_anchor(y_true,
                    anchors,
                    stride,
                    width=1,
                    height=1,
                    iou_thresh=0.25,
                    best_match_only=False, 
                    use_tie_breaker=True):
  """
  get the correct anchor that is assoiciated with each box using IOU
  
  Args:
    y_true: tf.Tensor[] for the list of bounding boxes in the yolo format
    anchors: list or tensor for the anchor boxes to be used in prediction
      found via Kmeans
    width: int for the image width
    height: int for the image height
  Return:
    tf.Tensor: y_true with the anchor associated with each ground truth
    box known
  """
  with tf.name_scope('get_best_anchor'):
    width = tf.cast(width, dtype=tf.float32)
    height = tf.cast(height, dtype=tf.float32)
    scaler = tf.convert_to_tensor([width, height])

    true_wh = tf.cast(y_true[..., 2:4], dtype=tf.float32) * scaler
    anchors = tf.cast(anchors, dtype=tf.float32)/stride

    k = tf.shape(anchors)[0]

    anchors = tf.concat([tf.zeros_like(anchors), anchors], axis=-1)
    truth_comp = tf.concat([tf.zeros_like(true_wh), true_wh], axis=-1)

    if iou_thresh >= 1.0:
      anchors = tf.expand_dims(anchors, axis=-2)
      truth_comp = tf.expand_dims(truth_comp, axis=-3)

      aspect = truth_comp[..., 2:4] / anchors[..., 2:4]
      aspect = tf.where(tf.math.is_nan(aspect), tf.zeros_like(aspect), aspect)
      aspect = tf.maximum(aspect, 1 / aspect)
      aspect = tf.where(tf.math.is_nan(aspect), tf.zeros_like(aspect), aspect)
      aspect = tf.reduce_max(aspect, axis=-1)

      values, indexes = tf.math.top_k(
          tf.transpose(-aspect, perm=[1, 0]),
          k=tf.cast(k, dtype=tf.int32),
          sorted=True)
      values = -values
      ind_mask = tf.cast(values < iou_thresh, dtype=indexes.dtype)
    else:
      # iou_raw = box_ops.compute_iou(truth_comp, anchors)
      truth_comp = box_ops.xcycwh_to_yxyx(truth_comp)
      anchors = box_ops.xcycwh_to_yxyx(anchors)
      iou_raw = box_ops.aggregated_comparitive_iou(
          truth_comp,
          anchors,
          iou_type=3,
      )
      values, indexes = tf.math.top_k(
          iou_raw,  #tf.transpose(iou_raw, perm=[0, 2, 1]),
          k=tf.cast(k, dtype=tf.int32),
          sorted=True)
      ind_mask = tf.cast(values >= iou_thresh, dtype=indexes.dtype)

    # pad the indexs such that all values less than the thresh are -1
    # add one, multiply the mask to zeros all the bad locations
    # subtract 1 makeing all the bad locations 0.
    if best_match_only:
      iou_index = ((indexes[..., 0:] + 1) * ind_mask[..., 0:]) - 1
    elif use_tie_breaker:
      iou_index = tf.concat([
          tf.expand_dims(indexes[..., 0], axis=-1),
          ((indexes[..., 1:] + 1) * ind_mask[..., 1:]) - 1], axis=-1)
    else:
      iou_index = tf.concat([
          tf.expand_dims(indexes[..., 0], axis=-1),
          tf.zeros_like(indexes[..., 1:]) - 1], axis=-1)

  return tf.cast(iou_index, dtype=tf.float32), tf.cast(values, dtype=tf.float32)

class YoloAnchorLabeler:
  def __init__(self, 
               anchors = None, 
               match_threshold = 0.25, 
               best_matches_only = False, 
               use_tie_breaker = True):
    self.anchors = anchors
    self.masks = self._get_mask()
    self.match_threshold = match_threshold
    self.best_matches_only = best_matches_only
    self.use_tie_breaker = use_tie_breaker

  def _get_mask(self):
    masks = {}
    start = 0

    minimum = int(min(self.anchors.keys()))
    maximum = int(max(self.anchors.keys()))
    for i in range(minimum, maximum + 1):
      per_scale = len(self.anchors[str(i)])
      masks[str(i)] = list(range(start, per_scale + start))
      start += per_scale
    return masks

  def _tie_breaking_search(self, anchors, mask, boxes, classes):
    mask = tf.cast(tf.reshape(mask, [1, 1, 1, -1]), anchors.dtype)
    anchors = tf.expand_dims(anchors, axis=-1)
    viable =  tf.where(tf.squeeze(anchors == mask, axis = 0))

    gather_id, _, anchor_id = tf.split(viable, 3, axis = -1)

    boxes = tf.gather_nd(boxes, gather_id)
    classes = tf.gather_nd(classes, gather_id)
    
    classes = tf.expand_dims(classes, axis = -1)
    classes = tf.cast(classes, boxes.dtype)
    anchor_id = tf.cast(anchor_id, boxes.dtype)
    return boxes, classes, anchor_id

  def _get_anchor_id(self, key, boxes, classes, anchors, width, height, stride):
    """Find the object anchor assignments in an anchor based paradigm. """
    
    # find the best anchor
    num_anchors = len(anchors)
    if self.best_matches_only:
      # get the best anchor for each box
      iou_index, _ = get_best_anchor(boxes, anchors, stride,
                                        width=width, height=height, 
                                        best_match_only=True, 
                                        iou_thresh=self.match_threshold)
      mask = range(num_anchors)
    else: 
      # stitch and search boxes across fpn levels
      anchorsvec = []
      for stitch in self.anchors.keys():
        anchorsvec.extend(self.anchors[stitch])

      # get the best anchor for each box
      iou_index, _ = get_best_anchor(boxes, anchorsvec, stride,
                                        width=width, height=height, 
                                        best_match_only=False, 
                                        use_tie_breaker=self.use_tie_breaker,
                                        iou_thresh=self.match_threshold)
      mask = self.masks[key]

    # search for the correct box to use
    (boxes, 
    classes,
    anchors) = self._tie_breaking_search(iou_index, mask, boxes, classes)
    return boxes, classes, anchors, num_anchors

  def _get_centers(self, boxes, classes, anchors, width, height, offset):
    """Find the object center assignments in an anchor based paradigm. """
    grid_xy, wh = tf.split(boxes, 2, axis = -1)
    wh_scale = tf.cast(tf.convert_to_tensor([width, height]), boxes.dtype)

    grid_xy = grid_xy * wh_scale
    centers = tf.math.floor(grid_xy)

    if offset != 0.0:   
      clamp = lambda x, ma: tf.maximum(
          tf.minimum(x, tf.cast(ma, x.dtype)), tf.zeros_like(x))

      grid_xy_index = grid_xy - centers
      positive_shift = ((grid_xy_index < offset) & (grid_xy > 1.))
      negative_shift = (
        (grid_xy_index > (1 - offset)) & (grid_xy < (wh_scale - 1.)))

      zero , _ = tf.split(tf.ones_like(positive_shift), 2, axis = -1)
      shift_mask = tf.concat(
        [zero, positive_shift, negative_shift], axis = -1)
      offset = tf.cast([[0, 0], [1, 0], 
                        [0, 1], [-1, 0], 
                        [0, -1]], offset.dtype) * offset

      num_shifts = tf.shape(shift_mask)
      num_shifts = num_shifts[-1]
      boxes = tf.tile(tf.expand_dims(boxes, axis = -2), [1, num_shifts, 1])
      classes = tf.tile(tf.expand_dims(classes, axis = -2), [1, num_shifts, 1])
      anchors = tf.tile(tf.expand_dims(anchors, axis = -2), [1, num_shifts, 1])

      shift_mask = tf.cast(shift_mask, boxes.dtype)
      shift_ind = shift_mask * tf.range(0, num_shifts, dtype = boxes.dtype) 
      shift_ind = shift_ind - (1 - shift_mask)
      shift_ind = tf.expand_dims(shift_ind, axis = -1)

      boxes_and_centers = tf.concat(
        [boxes, classes, anchors, shift_ind], axis = -1)
      boxes_and_centers = tf.reshape(boxes_and_centers, [-1, 7])
      _, center_ids =  tf.split(boxes_and_centers, [6, 1], axis = -1)

      #center_ids = tf.squeeze(center_ids, axis = -1)
      select = tf.where(center_ids >= 0)
      select, _ = tf.split(select, 2, axis = -1)

      boxes_and_centers = tf.gather_nd(boxes_and_centers, select)

      # center_ids = tf.cast(center_ids, tf.int32)
      center_ids = tf.gather_nd(center_ids, select)
      center_ids = tf.cast(center_ids, tf.int32)
      shifts = tf.gather_nd(offset, center_ids)

      boxes, classes, anchors, _ = tf.split(boxes_and_centers, 
                                                [4, 1, 1, 1], axis = -1)
      grid_xy, _ = tf.split(boxes, 2, axis = -1)
      centers = tf.math.floor(grid_xy * wh_scale - shifts)
      centers = clamp(centers, wh_scale - 1)
    
    x, y = tf.split(centers, 2, axis = -1)
    centers = tf.cast(tf.concat([y, x, anchors], axis = -1), tf.int32)
    return boxes, classes, centers

  def _get_anchor_free(self,
                       boxes,
                       classes,
                       height,
                       width,
                       stride,
                       fpn_limits,
                       center_radius=2.5):
    """Find the box assignements in an anchor free paradigm. """
    gen = loss_utils.GridGenerator(
      masks=None, anchors=[[1, 1]], scale_anchors=stride)
    grid_points = gen(width, height, 1, boxes.dtype)[0]
    grid_points = tf.squeeze(grid_points, axis=0)
    box_list = boxes
    class_list = classes

    grid_points = (grid_points + 0.5) * stride
    x_centers, y_centers = grid_points[..., 0], grid_points[..., 1]
    boxes *= (tf.convert_to_tensor([width, height, width, height]) * stride)

    tlbr_boxes = box_ops.xcycwh_to_yxyx(boxes)

    boxes = tf.reshape(boxes, [1, 1, -1, 4])
    tlbr_boxes = tf.reshape(tlbr_boxes, [1, 1, -1, 4])
    if self.use_tie_breaker:
      area = tf.reduce_prod(boxes[..., 2:], axis = -1)

    # check if the box is in the receptive feild of the this fpn level
    b_t = y_centers - tlbr_boxes[..., 0]
    b_l = x_centers - tlbr_boxes[..., 1]
    b_b = tlbr_boxes[..., 2] - y_centers
    b_r = tlbr_boxes[..., 3] - x_centers
    box_delta = tf.stack([b_t, b_l, b_b, b_r], axis=-1)
    if fpn_limits is not None:
      max_reg_targets_per_im = tf.reduce_max(box_delta, axis=-1)
      gt_min = max_reg_targets_per_im >= fpn_limits[0]
      gt_max = max_reg_targets_per_im <= fpn_limits[1]
      is_in_boxes = tf.logical_and(gt_min, gt_max)
    else:
      is_in_boxes = tf.reduce_min(box_delta, axis=-1) > 0.0
    is_in_boxes_all = tf.reduce_any(is_in_boxes, axis=(0, 1), keepdims=True)

    # check if the center is in the receptive feild of the this fpn level
    c_t = y_centers - (boxes[..., 1] - center_radius * stride)
    c_l = x_centers - (boxes[..., 0] - center_radius * stride)
    c_b = (boxes[..., 1] + center_radius * stride) - y_centers
    c_r = (boxes[..., 0] + center_radius * stride) - x_centers
    centers_delta = tf.stack([c_t, c_l, c_b, c_r], axis=-1)
    is_in_centers = tf.reduce_min(centers_delta, axis=-1) > 0.0
    is_in_centers_all = tf.reduce_any(is_in_centers, axis=(0, 1), keepdims=True)

    # colate all masks to get the final locations
    is_in_index = tf.logical_or(is_in_boxes_all, is_in_centers_all)
    is_in_boxes_and_center = tf.logical_and(is_in_boxes, is_in_centers)
    is_in_boxes_and_center = tf.logical_and(is_in_index, is_in_boxes_and_center)

    if self.use_tie_breaker:
      inf = 10000000
      boxes_all = tf.cast(is_in_boxes_and_center, area.dtype) 
      boxes_all = ((boxes_all * area) + ((1 - boxes_all) * inf))
      boxes_min = tf.reduce_min(boxes_all, axis = -1, keepdims = True)
      boxes_min = tf.where(boxes_min == inf, -1.0, boxes_min)
      is_in_boxes_and_center = boxes_all == boxes_min

    # construct the index update grid
    reps = tf.reduce_sum(tf.cast(is_in_boxes_and_center, tf.int16), axis=-1)
    indexes = tf.cast(tf.where(is_in_boxes_and_center), tf.int32)
    y, x, t = tf.split(indexes, 3, axis=-1)

    boxes = tf.gather_nd(box_list, t)
    classes = tf.cast(tf.gather_nd(class_list, t), boxes.dtype)
    reps = tf.gather_nd(reps, tf.concat([y, x], axis=-1))
    reps = tf.cast(tf.expand_dims(reps, axis=-1), boxes.dtype)
    classes = tf.cast(tf.expand_dims(classes, axis=-1), boxes.dtype)
    conf = tf.ones_like(classes)

    # return the samples and the indexes
    samples = tf.concat([boxes, conf, classes], axis=-1)
    indexes = tf.concat([y, x, tf.zeros_like(t)], axis=-1)
    return indexes, samples

  def __call__(self, 
               key, 
               boxes, 
               classes, 
               anchors, 
               width, 
               height, 
               stride, 
               scale_xy, 
               num_instances, 
               fpn_limits = None):
    """Builds the labels for a single image, not functional in batch mode. 
    
    Args: 
      boxes: `Tensor` of shape [None, 4] indicating the object locations in 
        an image. 
      classes: `Tensor` of shape [None] indicating the each objects classes.
      anchors: `List[List[int, float]]` representing the anchor boxes to build 
        the model against. 
      width: `int` for the images width. 
      height: `int` for the images height.
      stride: `int` for how much the image gets scaled at this level.
      scale_xy: `float` for the center shifts to apply when finding center 
        assignments for a box. 
      num_instances: `int` for the maximum number of expanded boxes to allow. 
      fpn_limits: `List[int]` given no anchor boxes this is used to limit the 
        boxes assied to the each fpn level based on the levels receptive feild. 

    Returns: 
      centers: `Tensor` of shape [None, 3] of indexes in the final grid where 
        boxes are located.  
      updates: `Tensor` of shape [None, 8] the value to place in the final grid.
      full: `Tensor` of [width/stride, height/stride, num_anchors, 1] holding 
        a mask of where boxes are locates for confidence losses. 
    """
    boxes = box_ops.yxyx_to_xcycwh(boxes)

    width //= stride
    height //= stride
    width = tf.cast(width, boxes.dtype)
    height = tf.cast(height, boxes.dtype)

    if fpn_limits is None:
      offset = tf.cast(0.5 * (scale_xy - 1), boxes.dtype)
      (boxes, classes, 
       anchors, num_anchors) = self._get_anchor_id(key, boxes, classes, anchors, 
                                                   width, height, stride)
      boxes, classes, centers = self._get_centers(boxes, classes, anchors, 
                                                  width, height, offset)
      ind_mask = tf.ones_like(classes)
      updates = tf.concat([boxes, ind_mask, classes], axis = -1)
    else:
      (centers, updates) = self._get_anchor_free(boxes, classes, height, 
                                                   width, stride, fpn_limits)
      boxes, ind_mask, classes = tf.split(updates, [4, 1, 1], axis = -1)
      num_anchors = 1


    width = tf.cast(width, tf.int32)
    height = tf.cast(height, tf.int32)
    full = tf.zeros([height, width, num_anchors, 1], dtype=classes.dtype)
    full = tf.tensor_scatter_nd_add(full, centers, ind_mask)
    centers = preprocessing_ops.pad_max_instances(
      centers, int(num_instances), pad_value=0, pad_axis=0)
    updates = preprocessing_ops.pad_max_instances(
      updates, int(num_instances), pad_value=0, pad_axis=0)
    return centers, updates, full

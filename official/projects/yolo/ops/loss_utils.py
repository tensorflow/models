# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Yolo loss utility functions."""

import numpy as np
import tensorflow as tf, tf_keras

from official.projects.yolo.ops import box_ops
from official.projects.yolo.ops import math_ops


@tf.custom_gradient
def sigmoid_bce(y, x_prime, label_smoothing):
  """Applies the Sigmoid Cross Entropy Loss.

  Implements the same derivative as that found in the Darknet C library.
  The derivative of this method is not the same as the standard binary cross
  entropy with logits function.

  The BCE with logits function equation is as follows:
    x = 1 / (1 + exp(-x_prime))
    bce = -ylog(x) - (1 - y)log(1 - x)

  The standard BCE with logits function derivative is as follows:
    dloss = -y/x + (1-y)/(1-x)
    dsigmoid = x * (1 - x)
    dx = dloss * dsigmoid

  This derivative can be reduced simply to:
    dx = (-y + x)

  This simplification is used by the darknet library in order to improve
  training stability. The gradient is almost the same
  as tf_keras.losses.binary_crossentropy but varies slightly and
  yields different performance.

  Args:
    y: `Tensor` holding ground truth data.
    x_prime: `Tensor` holding the predictions prior to application of the
      sigmoid operation.
    label_smoothing: float value between 0.0 and 1.0 indicating the amount of
      smoothing to apply to the data.

  Returns:
    bce: Tensor of the be applied loss values.
    delta: callable function indicating the custom gradient for this operation.
  """

  eps = 1e-9
  x = tf.math.sigmoid(x_prime)
  y = tf.stop_gradient(y * (1 - label_smoothing) + 0.5 * label_smoothing)
  bce = -y * tf.math.log(x + eps) - (1 - y) * tf.math.log(1 - x + eps)

  def delta(dpass):
    x = tf.math.sigmoid(x_prime)
    dx = (-y + x) * dpass
    dy = tf.zeros_like(y)
    return dy, dx, 0.0

  return bce, delta


def apply_mask(mask, x, value=0):
  """This function is used for gradient masking.

  The YOLO loss function makes extensive use of dynamically shaped tensors.
  To allow this use case on the TPU while preserving the gradient correctly
  for back propagation we use this masking function to use a tf.where operation
  to hard set masked location to have a gradient and a value of zero.

  Args:
    mask: A `Tensor` with the same shape as x used to select values of
      importance.
    x: A `Tensor` with the same shape as mask that will be getting masked.
    value: `float` constant additive value.

  Returns:
    x: A masked `Tensor` with the same shape as x.
  """
  mask = tf.cast(mask, tf.bool)
  masked = tf.where(mask, x, tf.zeros_like(x) + value)
  return masked


def build_grid(indexes, truths, preds, ind_mask, update=False, grid=None):
  """This function is used to broadcast elements into the output shape.

  This function is used to broadcasts a list of truths into the correct index
  in the output shape. This is used for the ground truth map construction in
  the scaled loss and the classification map in the darknet loss.

  Args:
    indexes: A `Tensor` for the indexes
    truths: A `Tensor` for the ground truth.
    preds: A `Tensor` for the predictions.
    ind_mask: A `Tensor` for the index masks.
    update: A `bool` for updating the grid.
    grid: A `Tensor` for the grid.

  Returns:
    grid: A `Tensor` representing the augmented grid.
  """
  # this function is used to broadcast all the indexes to the correct
  # into the correct ground truth mask, used for iou detection map
  # in the scaled loss and the classification mask in the darknet loss
  num_flatten = tf.shape(preds)[-1]

  # is there a way to verify that we are not on the CPU?
  ind_mask = tf.cast(ind_mask, indexes.dtype)

  # find all the batch indexes using the cumulated sum of a ones tensor
  # cumsum(ones) - 1 yeild the zero indexed batches
  bhep = tf.reduce_max(tf.ones_like(indexes), axis=-1, keepdims=True)
  bhep = tf.math.cumsum(bhep, axis=0) - 1

  # concatnate the batch sizes to the indexes
  indexes = tf.concat([bhep, indexes], axis=-1)
  indexes = apply_mask(tf.cast(ind_mask, indexes.dtype), indexes)
  indexes = (indexes + (ind_mask - 1))

  # mask truths
  truths = apply_mask(tf.cast(ind_mask, truths.dtype), truths)
  truths = (truths + (tf.cast(ind_mask, truths.dtype) - 1))

  # reshape the indexes into the correct shape for the loss,
  # just flatten all indexes but the last
  indexes = tf.reshape(indexes, [-1, 4])

  # also flatten the ground truth value on all axis but the last
  truths = tf.reshape(truths, [-1, num_flatten])

  # build a zero grid in the samve shape as the predicitons
  if grid is None:
    grid = tf.zeros_like(preds)
  # remove invalid values from the truths that may have
  # come up from computation, invalid = nan and inf
  truths = math_ops.rm_nan_inf(truths)

  # scatter update the zero grid
  if update:
    grid = tf.tensor_scatter_nd_update(grid, indexes, truths)
  else:
    grid = tf.tensor_scatter_nd_max(grid, indexes, truths)

  # stop gradient and return to avoid TPU errors and save compute
  # resources
  return grid


class GridGenerator:
  """Grid generator that generates anchor grids for box decoding."""

  def __init__(self, anchors, scale_anchors=None):
    """Initialize Grid Generator.

    Args:
      anchors: A `List[List[int]]` for the anchor boxes that are used in the
        model at all levels.
      scale_anchors: An `int` for how much to scale this level to get the
        original input shape.
    """
    self.dtype = tf_keras.backend.floatx()
    self._scale_anchors = scale_anchors
    self._anchors = tf.convert_to_tensor(anchors)
    return

  def _build_grid_points(self, lheight, lwidth, anchors, dtype):
    """Generate a grid of fixed grid edges for box center decoding."""
    with tf.name_scope('center_grid'):
      y = tf.range(0, lheight)
      x = tf.range(0, lwidth)
      x_left = tf.tile(
          tf.transpose(tf.expand_dims(x, axis=-1), perm=[1, 0]), [lheight, 1])
      y_left = tf.tile(tf.expand_dims(y, axis=-1), [1, lwidth])
      x_y = tf.stack([x_left, y_left], axis=-1)
      x_y = tf.cast(x_y, dtype=dtype)
      num = tf.shape(anchors)[0]
      x_y = tf.expand_dims(
          tf.tile(tf.expand_dims(x_y, axis=-2), [1, 1, num, 1]), axis=0)
    return x_y

  def _build_anchor_grid(self, height, width, anchors, dtype):
    """Get the transformed anchor boxes for each dimention."""
    with tf.name_scope('anchor_grid'):
      num = tf.shape(anchors)[0]
      anchors = tf.cast(anchors, dtype=dtype)
      anchors = tf.reshape(anchors, [1, 1, 1, num, 2])
      anchors = tf.tile(anchors, [1, tf.cast(height, tf.int32),
                                  tf.cast(width, tf.int32), 1, 1])
    return anchors

  def _extend_batch(self, grid, batch_size):
    return tf.tile(grid, [batch_size, 1, 1, 1, 1])

  def __call__(self, height, width, batch_size, dtype=None):
    if dtype is None:
      self.dtype = tf_keras.backend.floatx()
    else:
      self.dtype = dtype
    grid_points = self._build_grid_points(height, width, self._anchors,
                                          self.dtype)
    anchor_grid = self._build_anchor_grid(
        height, width,
        tf.cast(self._anchors, self.dtype) /
        tf.cast(self._scale_anchors, self.dtype), self.dtype)

    grid_points = self._extend_batch(grid_points, batch_size)
    anchor_grid = self._extend_batch(anchor_grid, batch_size)
    return grid_points, anchor_grid


TILE_SIZE = 50


class PairWiseSearch:
  """Apply a pairwise search between the ground truth and the labels.

  The goal is to indicate the locations where the predictions overlap with
  ground truth for dynamic ground truth associations.
  """

  def __init__(self,
               iou_type='iou',
               any_match=True,
               min_conf=0.0,
               track_boxes=False,
               track_classes=False):
    """Initialization of Pair Wise Search.

    Args:
      iou_type: An `str` for the iou type to use.
      any_match: A `bool` for any match(no class match).
      min_conf: An `int` for minimum confidence threshold.
      track_boxes: A `bool` dynamic box assignment.
      track_classes: A `bool` dynamic class assignment.
    """
    self.iou_type = iou_type
    self._any = any_match
    self._min_conf = min_conf
    self._track_boxes = track_boxes
    self._track_classes = track_classes
    return

  def box_iou(self, true_box, pred_box):
    # based on the type of loss, compute the iou loss for a box
    # compute_<name> indicated the type of iou to use
    if self.iou_type == 'giou':
      _, iou = box_ops.compute_giou(true_box, pred_box)
    elif self.iou_type == 'ciou':
      _, iou = box_ops.compute_ciou(true_box, pred_box)
    else:
      iou = box_ops.compute_iou(true_box, pred_box)
    return iou

  def _search_body(self, pred_box, pred_class, boxes, classes, running_boxes,
                   running_classes, max_iou, idx):
    """Main search fn."""

    # capture the batch size to be used, and gather a slice of
    # boxes from the ground truth. currently TILE_SIZE = 50, to
    # save memory
    batch_size = tf.shape(boxes)[0]
    box_slice = tf.slice(boxes, [0, idx * TILE_SIZE, 0],
                         [batch_size, TILE_SIZE, 4])

    # match the dimentions of the slice to the model predictions
    # shape: [batch_size, 1, 1, num, TILE_SIZE, 4]
    box_slice = tf.expand_dims(box_slice, axis=1)
    box_slice = tf.expand_dims(box_slice, axis=1)
    box_slice = tf.expand_dims(box_slice, axis=1)

    box_grid = tf.expand_dims(pred_box, axis=-2)

    # capture the classes
    class_slice = tf.slice(classes, [0, idx * TILE_SIZE],
                           [batch_size, TILE_SIZE])
    class_slice = tf.expand_dims(class_slice, axis=1)
    class_slice = tf.expand_dims(class_slice, axis=1)
    class_slice = tf.expand_dims(class_slice, axis=1)

    iou = self.box_iou(box_slice, box_grid)

    if self._min_conf > 0.0:
      if not self._any:
        class_grid = tf.expand_dims(pred_class, axis=-2)
        class_mask = tf.one_hot(
            tf.cast(class_slice, tf.int32),
            depth=tf.shape(pred_class)[-1],
            dtype=pred_class.dtype)
        class_mask = tf.reduce_any(tf.equal(class_mask, class_grid), axis=-1)
      else:
        class_mask = tf.reduce_max(pred_class, axis=-1, keepdims=True)
      class_mask = tf.cast(class_mask, iou.dtype)
      iou *= class_mask

    max_iou_ = tf.concat([max_iou, iou], axis=-1)
    max_iou = tf.reduce_max(max_iou_, axis=-1, keepdims=True)
    ind = tf.expand_dims(tf.argmax(max_iou_, axis=-1), axis=-1)

    if self._track_boxes:
      running_boxes = tf.expand_dims(running_boxes, axis=-2)
      box_slice = tf.zeros_like(running_boxes) + box_slice
      box_slice = tf.concat([running_boxes, box_slice], axis=-2)
      running_boxes = tf.gather_nd(box_slice, ind, batch_dims=4)

    if self._track_classes:
      running_classes = tf.expand_dims(running_classes, axis=-1)
      class_slice = tf.zeros_like(running_classes) + class_slice
      class_slice = tf.concat([running_classes, class_slice], axis=-1)
      running_classes = tf.gather_nd(class_slice, ind, batch_dims=4)

    return (pred_box, pred_class, boxes, classes, running_boxes,
            running_classes, max_iou, idx + 1)

  def __call__(self,
               pred_boxes,
               pred_classes,
               boxes,
               classes,
               clip_thresh=0.0):
    num_boxes = tf.shape(boxes)[-2]
    num_tiles = (num_boxes // TILE_SIZE) - 1

    if self._min_conf > 0.0:
      pred_classes = tf.cast(pred_classes > self._min_conf, pred_classes.dtype)

    def _loop_cond(unused_pred_box, unused_pred_class, boxes, unused_classes,
                   unused_running_boxes, unused_running_classes, unused_max_iou,
                   idx):

      # check that the slice has boxes that all zeros
      batch_size = tf.shape(boxes)[0]
      box_slice = tf.slice(boxes, [0, idx * TILE_SIZE, 0],
                           [batch_size, TILE_SIZE, 4])

      return tf.logical_and(idx < num_tiles,
                            tf.math.greater(tf.reduce_sum(box_slice), 0))

    running_boxes = tf.zeros_like(pred_boxes)
    running_classes = tf.zeros_like(tf.reduce_sum(running_boxes, axis=-1))
    max_iou = tf.zeros_like(tf.reduce_sum(running_boxes, axis=-1))
    max_iou = tf.expand_dims(max_iou, axis=-1)

    (pred_boxes, pred_classes, boxes, classes, running_boxes, running_classes,
     max_iou, _) = tf.while_loop(_loop_cond, self._search_body, [
         pred_boxes, pred_classes, boxes, classes, running_boxes,
         running_classes, max_iou,
         tf.constant(0)
     ])

    mask = tf.cast(max_iou > clip_thresh, running_boxes.dtype)
    running_boxes *= mask
    running_classes *= tf.squeeze(mask, axis=-1)
    max_iou *= mask
    max_iou = tf.squeeze(max_iou, axis=-1)
    mask = tf.squeeze(mask, axis=-1)

    return (tf.stop_gradient(running_boxes), tf.stop_gradient(running_classes),
            tf.stop_gradient(max_iou), tf.stop_gradient(mask))


def average_iou(iou):
  """Computes the average intersection over union without counting locations.

  where the iou is zero.

  Args:
    iou: A `Tensor` representing the iou values.

  Returns:
    tf.stop_gradient(avg_iou): A `Tensor` representing average
     intersection over union.
  """
  iou_sum = tf.reduce_sum(iou, axis=tf.range(1, tf.shape(tf.shape(iou))[0]))
  counts = tf.cast(
      tf.math.count_nonzero(iou, axis=tf.range(1,
                                               tf.shape(tf.shape(iou))[0])),
      iou.dtype)
  avg_iou = tf.reduce_mean(math_ops.divide_no_nan(iou_sum, counts))
  return tf.stop_gradient(avg_iou)


def _scale_boxes(encoded_boxes, width, height, anchor_grid, grid_points,
                 scale_xy):
  """Decodes models boxes applying and exponential to width and height maps."""
  # split the boxes
  pred_xy = encoded_boxes[..., 0:2]
  pred_wh = encoded_boxes[..., 2:4]

  # build a scaling tensor to get the offset of th ebox relative to the image
  scaler = tf.convert_to_tensor([height, width, height, width])
  scale_xy = tf.cast(scale_xy, encoded_boxes.dtype)

  # apply the sigmoid
  pred_xy = tf.math.sigmoid(pred_xy)

  # scale the centers and find the offset of each box relative to
  # their center pixel
  pred_xy = pred_xy * scale_xy - 0.5 * (scale_xy - 1)

  # scale the offsets and add them to the grid points or a tensor that is
  # the realtive location of each pixel
  box_xy = grid_points + pred_xy

  # scale the width and height of the predictions and corlate them
  # to anchor boxes
  box_wh = tf.math.exp(pred_wh) * anchor_grid

  # build the final predicted box
  scaled_box = tf.concat([box_xy, box_wh], axis=-1)
  pred_box = scaled_box / scaler

  # shift scaled boxes
  scaled_box = tf.concat([pred_xy, box_wh], axis=-1)
  return (scaler, scaled_box, pred_box)


@tf.custom_gradient
def _darknet_boxes(encoded_boxes, width, height, anchor_grid, grid_points,
                   max_delta, scale_xy):
  """Wrapper for _scale_boxes to implement a custom gradient."""
  (scaler, scaled_box, pred_box) = _scale_boxes(encoded_boxes, width, height,
                                                anchor_grid, grid_points,
                                                scale_xy)

  def delta(unused_dy_scaler, dy_scaled, dy):
    dy_xy, dy_wh = tf.split(dy, 2, axis=-1)
    dy_xy_, dy_wh_ = tf.split(dy_scaled, 2, axis=-1)

    # add all the gradients that may have been applied to the
    # boxes and those that have been applied to the width and height
    dy_wh += dy_wh_
    dy_xy += dy_xy_

    # propagate the exponential applied to the width and height in
    # order to ensure the gradient propagated is of the correct
    # magnitude
    pred_wh = encoded_boxes[..., 2:4]
    dy_wh *= tf.math.exp(pred_wh)

    dbox = tf.concat([dy_xy, dy_wh], axis=-1)

    # apply the gradient clipping to xy and wh
    dbox = math_ops.rm_nan_inf(dbox)
    delta = tf.cast(max_delta, dbox.dtype)
    dbox = tf.clip_by_value(dbox, -delta, delta)
    return dbox, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

  return (scaler, scaled_box, pred_box), delta


def _new_coord_scale_boxes(encoded_boxes, width, height, anchor_grid,
                           grid_points, scale_xy):
  """Decodes models boxes by squaring and scaling the width and height maps."""
  # split the boxes
  pred_xy = encoded_boxes[..., 0:2]
  pred_wh = encoded_boxes[..., 2:4]

  # build a scaling tensor to get the offset of th ebox relative to the image
  scaler = tf.convert_to_tensor([height, width, height, width])
  scale_xy = tf.cast(scale_xy, pred_xy.dtype)

  # apply the sigmoid
  pred_xy = tf.math.sigmoid(pred_xy)
  pred_wh = tf.math.sigmoid(pred_wh)

  # scale the xy offset predictions according to the config
  pred_xy = pred_xy * scale_xy - 0.5 * (scale_xy - 1)

  # find the true offset from the grid points and the scaler
  # where the grid points are the relative offset of each pixel with
  # in the image
  box_xy = grid_points + pred_xy

  # decode the widht and height of the boxes and correlate them
  # to the anchor boxes
  box_wh = (2 * pred_wh)**2 * anchor_grid

  # build the final boxes
  scaled_box = tf.concat([box_xy, box_wh], axis=-1)
  pred_box = scaled_box / scaler

  # shift scaled boxes
  scaled_box = tf.concat([pred_xy, box_wh], axis=-1)
  return (scaler, scaled_box, pred_box)


@tf.custom_gradient
def _darknet_new_coord_boxes(encoded_boxes, width, height, anchor_grid,
                             grid_points, max_delta, scale_xy):
  """Wrapper for _new_coord_scale_boxes to implement a custom gradient."""
  (scaler, scaled_box,
   pred_box) = _new_coord_scale_boxes(encoded_boxes, width, height, anchor_grid,
                                      grid_points, scale_xy)

  def delta(unused_dy_scaler, dy_scaled, dy):
    dy_xy, dy_wh = tf.split(dy, 2, axis=-1)
    dy_xy_, dy_wh_ = tf.split(dy_scaled, 2, axis=-1)

    # add all the gradients that may have been applied to the
    # boxes and those that have been applied to the width and height
    dy_wh += dy_wh_
    dy_xy += dy_xy_

    dbox = tf.concat([dy_xy, dy_wh], axis=-1)

    # apply the gradient clipping to xy and wh
    dbox = math_ops.rm_nan_inf(dbox)
    delta = tf.cast(max_delta, dbox.dtype)
    dbox = tf.clip_by_value(dbox, -delta, delta)
    return dbox, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

  return (scaler, scaled_box, pred_box), delta


def _anchor_free_scale_boxes(encoded_boxes,
                             width,
                             height,
                             stride,
                             grid_points,
                             darknet=False):
  """Decode models boxes using FPN stride under anchor free conditions."""
  del darknet
  # split the boxes
  pred_xy = encoded_boxes[..., 0:2]
  pred_wh = encoded_boxes[..., 2:4]

  # build a scaling tensor to get the offset of th ebox relative to the image
  scaler = tf.convert_to_tensor([height, width, height, width])

  # scale the offsets and add them to the grid points or a tensor that is
  # the realtive location of each pixel
  box_xy = (grid_points + pred_xy)

  # scale the width and height of the predictions and corlate them
  # to anchor boxes
  box_wh = tf.math.exp(pred_wh)

  # build the final predicted box
  scaled_box = tf.concat([box_xy, box_wh], axis=-1)

  # properly scaling boxes gradeints
  scaled_box = scaled_box * tf.cast(stride, scaled_box.dtype)
  pred_box = scaled_box / tf.cast(scaler * stride, scaled_box.dtype)
  return (scaler, scaled_box, pred_box)


def get_predicted_box(width,
                      height,
                      encoded_boxes,
                      anchor_grid,
                      grid_points,
                      scale_xy,
                      stride,
                      darknet=False,
                      box_type='original',
                      max_delta=np.inf):
  """Decodes the predicted boxes from the model format to a usable format.

  This function decodes the model outputs into the [x, y, w, h] format for
  use in the loss function as well as for use within the detection generator.

  Args:
    width: A `float` scalar indicating the width of the prediction layer.
    height: A `float` scalar indicating the height of the prediction layer
    encoded_boxes: A `Tensor` of shape [..., height, width, 4] holding encoded
      boxes.
    anchor_grid: A `Tensor` of shape [..., 1, 1, 2] holding the anchor boxes
      organized for box decoding, box width and height.
    grid_points: A `Tensor` of shape [..., height, width, 2] holding the anchor
      boxes for decoding the box centers.
    scale_xy: A `float` scaler used to indicate the range for each center
      outside of its given [..., i, j, 4] index, where i and j are indexing
      pixels along the width and height of the predicted output map.
    stride: An `int` defining the amount of down stride realtive to the input
      image.
    darknet: A `bool` used to select between custom gradient and default
      autograd.
    box_type: An `str` indicating the type of box encoding that is being used.
    max_delta: A `float` scaler used for gradient clipping in back propagation.

  Returns:
    scaler: A `Tensor` of shape [4] returned to allow the scaling of the ground
      truth boxes to be of the same magnitude as the decoded predicted boxes.
    scaled_box: A `Tensor` of shape [..., height, width, 4] with the predicted
      boxes.
    pred_box: A `Tensor` of shape [..., height, width, 4] with the predicted
      boxes divided by the scaler parameter used to put all boxes in the [0, 1]
      range.
  """
  if box_type == 'anchor_free':
    (scaler, scaled_box, pred_box) = _anchor_free_scale_boxes(
        encoded_boxes, width, height, stride, grid_points, darknet=darknet)
  elif darknet:

    # pylint:disable=unbalanced-tuple-unpacking
    # if we are using the darknet loss we shoud nto propagate the
    # decoding of the box
    if box_type == 'scaled':
      (scaler, scaled_box,
       pred_box) = _darknet_new_coord_boxes(encoded_boxes, width, height,
                                            anchor_grid, grid_points, max_delta,
                                            scale_xy)
    else:
      (scaler, scaled_box,
       pred_box) = _darknet_boxes(encoded_boxes, width, height, anchor_grid,
                                  grid_points, max_delta, scale_xy)
  else:
    # if we are using the scaled loss we should propagate the decoding of
    # the boxes
    if box_type == 'scaled':
      (scaler, scaled_box,
       pred_box) = _new_coord_scale_boxes(encoded_boxes, width, height,
                                          anchor_grid, grid_points, scale_xy)
    else:
      (scaler, scaled_box, pred_box) = _scale_boxes(encoded_boxes, width,
                                                    height, anchor_grid,
                                                    grid_points, scale_xy)

  return (scaler, scaled_box, pred_box)

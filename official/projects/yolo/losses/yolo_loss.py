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

"""Yolo Loss function."""
import abc
import collections
import functools

import tensorflow as tf

from official.projects.yolo.ops import box_ops
from official.projects.yolo.ops import loss_utils
from official.projects.yolo.ops import math_ops


class YoloLossBase(object, metaclass=abc.ABCMeta):
  """Parameters for the YOLO loss functions used at each detection generator.

  This base class implements the base functionality required to implement a Yolo
  Loss function.
  """

  def __init__(self,
               classes,
               anchors,
               path_stride=1,
               ignore_thresh=0.7,
               truth_thresh=1.0,
               loss_type='ciou',
               iou_normalizer=1.0,
               cls_normalizer=1.0,
               object_normalizer=1.0,
               label_smoothing=0.0,
               objectness_smooth=True,
               update_on_repeat=False,
               box_type='original',
               scale_x_y=1.0,
               max_delta=10):
    """Loss Function Initialization.

    Args:
      classes: `int` for the number of classes
      anchors: `List[List[int]]` for the anchor boxes that are used in the model
        at all levels. For anchor free prediction set the anchor list to be the
        same as the image resolution.
      path_stride: `int` for how much to scale this level to get the orginal
        input shape.
      ignore_thresh: `float` for the IOU value over which the loss is not
        propagated, and a detection is assumed to have been made.
      truth_thresh: `float` for the IOU value over which the loss is propagated
        despite a detection being made.
      loss_type: `str` for the typeof iou loss to use with in {ciou, diou, giou,
        iou}.
      iou_normalizer: `float` for how much to scale the loss on the IOU or the
        boxes.
      cls_normalizer: `float` for how much to scale the loss on the classes.
      object_normalizer: `float` for how much to scale loss on the detection
        map.
      label_smoothing: `float` for how much to smooth the loss on the classes.
      objectness_smooth: `float` for how much to smooth the loss on the
        detection map.
      update_on_repeat: `bool` for whether to replace with the newest or the
        best value when an index is consumed by multiple objects.
      box_type: `bool` for which scaling type to use.
      scale_x_y: dictionary `float` values inidcating how far each pixel can see
        outside of its containment of 1.0. a value of 1.2 indicates there is a
        20% extended radius around each pixel that this specific pixel can
        predict values for a center at. the center can range from 0 - value/2 to
        1 + value/2, this value is set in the yolo filter, and resused here.
        there should be one value for scale_xy for each level from min_level to
        max_level.
      max_delta: gradient clipping to apply to the box loss.
    """
    self._loss_type = loss_type
    self._classes = classes
    self._num = tf.cast(len(anchors), dtype=tf.int32)
    self._truth_thresh = truth_thresh
    self._ignore_thresh = ignore_thresh
    self._anchors = anchors

    self._iou_normalizer = iou_normalizer
    self._cls_normalizer = cls_normalizer
    self._object_normalizer = object_normalizer
    self._scale_x_y = scale_x_y
    self._max_delta = max_delta

    self._label_smoothing = tf.cast(label_smoothing, tf.float32)
    self._objectness_smooth = float(objectness_smooth)
    self._update_on_repeat = update_on_repeat
    self._box_type = box_type
    self._path_stride = path_stride

    box_kwargs = dict(
        stride=self._path_stride,
        scale_xy=self._scale_x_y,
        box_type=self._box_type,
        max_delta=self._max_delta)
    self._decode_boxes = functools.partial(
        loss_utils.get_predicted_box, **box_kwargs)

    self._search_pairs = lambda *args: (None, None, None, None)
    self._build_per_path_attributes()

  def box_loss(self, true_box, pred_box, darknet=False):
    """Call iou function and use it to compute the loss for the box maps."""
    if self._loss_type == 'giou':
      iou, liou = box_ops.compute_giou(true_box, pred_box)
    elif self._loss_type == 'ciou':
      iou, liou = box_ops.compute_ciou(true_box, pred_box, darknet=darknet)
    else:
      liou = iou = box_ops.compute_iou(true_box, pred_box)
    loss_box = 1 - liou
    return iou, liou, loss_box

  def _tiled_global_box_search(self,
                               pred_boxes,
                               pred_classes,
                               boxes,
                               classes,
                               true_conf,
                               smoothed,
                               scale=None):
    """Search of all groundtruths to associate groundtruths to predictions."""

    boxes = box_ops.yxyx_to_xcycwh(boxes)

    if scale is not None:
      boxes = boxes * tf.cast(tf.stop_gradient(scale), boxes.dtype)

    # Search all predictions against ground truths to find mathcing boxes for
    # each pixel.
    _, _, iou_max, _ = self._search_pairs(pred_boxes, pred_classes, boxes,
                                          classes)

    if iou_max is None:
      return true_conf, tf.ones_like(true_conf)

    # Find the exact indexes to ignore and keep.
    ignore_mask = tf.cast(iou_max < self._ignore_thresh, pred_boxes.dtype)
    iou_mask = iou_max > self._ignore_thresh

    if not smoothed:
      # Ignore all pixels where a box was not supposed to be predicted but a
      # high confidence box was predicted.
      obj_mask = true_conf + (1 - true_conf) * ignore_mask
    else:
      # Replace pixels in the tre confidence map with the max iou predicted
      # with in that cell.
      obj_mask = tf.ones_like(true_conf)
      iou_ = (1 - self._objectness_smooth) + self._objectness_smooth * iou_max
      iou_ = tf.where(iou_max > 0, iou_, tf.zeros_like(iou_))
      true_conf = tf.where(iou_mask, iou_, true_conf)

    # Stop gradient so while loop is not tracked.
    obj_mask = tf.stop_gradient(obj_mask)
    true_conf = tf.stop_gradient(true_conf)
    return true_conf, obj_mask

  def __call__(self, true_counts, inds, y_true, boxes, classes, y_pred):
    """Call function to compute the loss and a set of metrics per FPN level.

    Args:
      true_counts: `Tensor` of shape [batchsize, height, width, num_anchors]
        represeneting how many boxes are in a given pixel [j, i] in the output
        map.
      inds: `Tensor` of shape [batchsize, None, 3] indicating the location [j,
        i] that a given box is associatied with in the FPN prediction map.
      y_true: `Tensor` of shape [batchsize, None, 8] indicating the actual box
        associated with each index in the inds tensor list.
      boxes: `Tensor` of shape [batchsize, None, 4] indicating the original
        ground truth boxes for each image as they came from the decoder used for
        bounding box search.
      classes: `Tensor` of shape [batchsize, None, 1] indicating the original
        ground truth classes for each image as they came from the decoder used
        for bounding box search.
      y_pred: `Tensor` of shape [batchsize, height, width, output_depth] holding
        the models output at a specific FPN level.

    Returns:
      loss: `float` for the actual loss.
      box_loss: `float` loss on the boxes used for metrics.
      conf_loss: `float` loss on the confidence used for metrics.
      class_loss: `float` loss on the classes used for metrics.
      avg_iou: `float` metric for the average iou between predictions and ground
        truth.
      avg_obj: `float` metric for the average confidence of the model for
        predictions.
    """
    (loss, box_loss, conf_loss, class_loss, mean_loss, iou, pred_conf, ind_mask,
     grid_mask) = self._compute_loss(true_counts, inds, y_true, boxes, classes,
                                     y_pred)

    # Metric compute using done here to save time and resources.
    sigmoid_conf = tf.stop_gradient(tf.sigmoid(pred_conf))
    iou = tf.stop_gradient(iou)
    avg_iou = loss_utils.average_iou(
        loss_utils.apply_mask(tf.squeeze(ind_mask, axis=-1), iou))
    avg_obj = loss_utils.average_iou(
        tf.squeeze(sigmoid_conf, axis=-1) * grid_mask)
    return (loss, box_loss, conf_loss, class_loss, mean_loss,
            tf.stop_gradient(avg_iou), tf.stop_gradient(avg_obj))

  @abc.abstractmethod
  def _build_per_path_attributes(self):
    """Additional initialization required for each YOLO loss version."""
    ...

  @abc.abstractmethod
  def _compute_loss(self, true_counts, inds, y_true, boxes, classes, y_pred):
    """The actual logic to apply to the raw model for optimization."""
    ...

  def post_path_aggregation(self, loss, box_loss, conf_loss, class_loss,
                            ground_truths, predictions):  # pylint:disable=unused-argument
    """This method allows for post processing of a loss value.

    After the loss has been aggregated across all the FPN levels some post
    proceessing may need to occur to poroperly scale the loss. The default
    behavior is to pass the loss through with no alterations. Passing the
    individual losses for each mask will allow for aggeregation of loss across
    paths for some losses.

    Args:
      loss: `tf.float` scalar for the actual loss.
      box_loss: `tf.float` for the loss on the boxs only.
      conf_loss: `tf.float` for the loss on the confidences only.
      class_loss: `tf.float` for the loss on the classes only.
      ground_truths: `Dict` holding all the ground truth tensors.
      predictions: `Dict` holding all the predicted values.

    Returns:
      loss: `tf.float` scalar for the scaled loss.
      scale: `tf.float` how much the loss was scaled by.
    """
    del box_loss
    del conf_loss
    del class_loss
    del ground_truths
    del predictions
    return loss, tf.ones_like(loss)

  @abc.abstractmethod
  def cross_replica_aggregation(self, loss, num_replicas_in_sync):
    """This controls how the loss should be aggregated across replicas."""
    ...


@tf.custom_gradient
def grad_sigmoid(values):
  """This function scales the gradient as if a signmoid was applied.

  This is used in the Darknet Loss when the choosen box type is the scaled
  coordinate type. This function is used to match the propagated gradient to
  match that of the Darkent Yolov4 model. This is an Identity operation that
  allows us to add some extra steps to the back propagation.

  Args:
    values: A tensor of any shape.

  Returns:
    values: The unaltered input tensor.
    delta: A custom gradient function that adds the sigmoid step to the
      backpropagation.
  """

  def delta(dy):
    t = tf.math.sigmoid(values)
    return dy * t * (1 - t)

  return values, delta


class DarknetLoss(YoloLossBase):
  """This class implements the full logic for the standard Yolo models."""

  def _build_per_path_attributes(self):
    """Paramterization of pair wise search and grid generators.

    Objects created here are used for box decoding and dynamic ground truth
    association.
    """
    self._anchor_generator = loss_utils.GridGenerator(
        anchors=self._anchors,
        scale_anchors=self._path_stride)

    if self._ignore_thresh > 0.0:
      self._search_pairs = loss_utils.PairWiseSearch(
          iou_type='iou', any_match=True, min_conf=0.25)
    return

  def _compute_loss(self, true_counts, inds, y_true, boxes, classes, y_pred):
    """Per FPN path loss logic used for Yolov3, Yolov4, and Yolo-Tiny."""
    if self._box_type == 'scaled':
      # Darknet Model Propagates a sigmoid once in back prop so we replicate
      # that behaviour
      y_pred = grad_sigmoid(y_pred)

    # Generate and store constants and format output.
    shape = tf.shape(true_counts)
    batch_size, width, height, num = shape[0], shape[1], shape[2], shape[3]
    fwidth = tf.cast(width, tf.float32)
    fheight = tf.cast(height, tf.float32)
    grid_points, anchor_grid = self._anchor_generator(
        width, height, batch_size, dtype=tf.float32)

    # Cast all input compontnts to float32 and stop gradient to save memory.
    boxes = tf.stop_gradient(tf.cast(boxes, tf.float32))
    classes = tf.stop_gradient(tf.cast(classes, tf.float32))
    y_true = tf.stop_gradient(tf.cast(y_true, tf.float32))
    true_counts = tf.stop_gradient(tf.cast(true_counts, tf.float32))
    true_conf = tf.stop_gradient(tf.clip_by_value(true_counts, 0.0, 1.0))
    grid_points = tf.stop_gradient(grid_points)
    anchor_grid = tf.stop_gradient(anchor_grid)

    # Split all the ground truths to use as separate items in loss computation.
    (true_box, ind_mask, true_class) = tf.split(y_true, [4, 1, 1], axis=-1)
    true_conf = tf.squeeze(true_conf, axis=-1)
    true_class = tf.squeeze(true_class, axis=-1)
    grid_mask = true_conf

    # Splits all predictions.
    y_pred = tf.cast(
        tf.reshape(y_pred, [batch_size, width, height, num, -1]), tf.float32)
    pred_box, pred_conf, pred_class = tf.split(y_pred, [4, 1, -1], axis=-1)

    # Decode the boxes to be used for loss compute.
    _, _, pred_box = self._decode_boxes(
        fwidth, fheight, pred_box, anchor_grid, grid_points, darknet=True)

    # If the ignore threshold is enabled, search all boxes ignore all
    # IOU valeus larger than the ignore threshold that are not in the
    # noted ground truth list.
    if self._ignore_thresh != 0.0:
      (true_conf, obj_mask) = self._tiled_global_box_search(
          pred_box,
          tf.stop_gradient(tf.sigmoid(pred_class)),
          boxes,
          classes,
          true_conf,
          smoothed=self._objectness_smooth > 0)

    # Build the one hot class list that are used for class loss.
    true_class = tf.one_hot(
        tf.cast(true_class, tf.int32),
        depth=tf.shape(pred_class)[-1],
        dtype=pred_class.dtype)
    true_class = tf.stop_gradient(loss_utils.apply_mask(ind_mask, true_class))

    # Reorganize the one hot class list as a grid.
    true_class_grid = loss_utils.build_grid(
        inds, true_class, pred_class, ind_mask, update=False)
    true_class_grid = tf.stop_gradient(true_class_grid)

    # Use the class mask to find the number of objects located in
    # each predicted grid cell/pixel.
    counts = true_class_grid
    counts = tf.reduce_sum(counts, axis=-1, keepdims=True)
    reps = tf.gather_nd(counts, inds, batch_dims=1)
    reps = tf.squeeze(reps, axis=-1)
    reps = tf.stop_gradient(tf.where(reps == 0.0, tf.ones_like(reps), reps))

    # Compute the loss for only the cells in which the boxes are located.
    pred_box = loss_utils.apply_mask(ind_mask,
                                     tf.gather_nd(pred_box, inds, batch_dims=1))
    iou, _, box_loss = self.box_loss(true_box, pred_box, darknet=True)
    box_loss = loss_utils.apply_mask(tf.squeeze(ind_mask, axis=-1), box_loss)
    box_loss = math_ops.divide_no_nan(box_loss, reps)
    box_loss = tf.cast(tf.reduce_sum(box_loss, axis=1), dtype=y_pred.dtype)

    if self._update_on_repeat:
      # Converts list of gound truths into a grid where repeated values
      # are replaced by the most recent value. So some class identities may
      # get lost but the loss computation will be more stable. Results are
      # more consistent.

      # Compute the sigmoid binary cross entropy for the class maps.
      class_loss = tf.reduce_mean(
          loss_utils.sigmoid_bce(
              tf.expand_dims(true_class_grid, axis=-1),
              tf.expand_dims(pred_class, axis=-1), self._label_smoothing),
          axis=-1)

      # Apply normalization to the class losses.
      if self._cls_normalizer < 1.0:
        # Build a mask based on the true class locations.
        cls_norm_mask = true_class_grid
        # Apply the classes weight to class indexes were one_hot is one.
        class_loss *= ((1 - cls_norm_mask) +
                       cls_norm_mask * self._cls_normalizer)

      # Mask to the class loss and compute the sum over all the objects.
      class_loss = tf.reduce_sum(class_loss, axis=-1)
      class_loss = loss_utils.apply_mask(grid_mask, class_loss)
      class_loss = math_ops.rm_nan_inf(class_loss, val=0.0)
      class_loss = tf.cast(
          tf.reduce_sum(class_loss, axis=(1, 2, 3)), dtype=y_pred.dtype)
    else:
      # Computes the loss while keeping the structure as a list in
      # order to ensure all objects are considered. In some cases can
      # make training more unstable but may also return higher APs.
      pred_class = loss_utils.apply_mask(
          ind_mask, tf.gather_nd(pred_class, inds, batch_dims=1))
      class_loss = tf.keras.losses.binary_crossentropy(
          tf.expand_dims(true_class, axis=-1),
          tf.expand_dims(pred_class, axis=-1),
          label_smoothing=self._label_smoothing,
          from_logits=True)
      class_loss = loss_utils.apply_mask(ind_mask, class_loss)
      class_loss = math_ops.divide_no_nan(class_loss,
                                          tf.expand_dims(reps, axis=-1))
      class_loss = tf.cast(
          tf.reduce_sum(class_loss, axis=(1, 2)), dtype=y_pred.dtype)
      class_loss *= self._cls_normalizer

    # Compute the sigmoid binary cross entropy for the confidence maps.
    bce = tf.reduce_mean(
        loss_utils.sigmoid_bce(
            tf.expand_dims(true_conf, axis=-1), pred_conf, 0.0),
        axis=-1)

    # Mask the confidence loss and take the sum across all the grid cells.
    if self._ignore_thresh != 0.0:
      bce = loss_utils.apply_mask(obj_mask, bce)
    conf_loss = tf.cast(tf.reduce_sum(bce, axis=(1, 2, 3)), dtype=y_pred.dtype)

    # Apply the weights to each loss.
    box_loss *= self._iou_normalizer
    conf_loss *= self._object_normalizer

    # Add all the losses together then take the mean over the batches.
    loss = box_loss + class_loss + conf_loss
    loss = tf.reduce_mean(loss)

    # Reduce the mean of the losses to use as a metric.
    box_loss = tf.reduce_mean(box_loss)
    conf_loss = tf.reduce_mean(conf_loss)
    class_loss = tf.reduce_mean(class_loss)

    return (loss, box_loss, conf_loss, class_loss, loss, iou, pred_conf,
            ind_mask, grid_mask)

  def cross_replica_aggregation(self, loss, num_replicas_in_sync):
    """This method is not specific to each loss path, but each loss type."""
    return loss / num_replicas_in_sync


class ScaledLoss(YoloLossBase):
  """This class implements the full logic for the scaled Yolo models."""

  def _build_per_path_attributes(self):
    """Paramterization of pair wise search and grid generators.

    Objects created here are used for box decoding and dynamic ground truth
    association.
    """
    self._anchor_generator = loss_utils.GridGenerator(
        anchors=self._anchors,
        scale_anchors=self._path_stride)

    if self._ignore_thresh > 0.0:
      self._search_pairs = loss_utils.PairWiseSearch(
          iou_type=self._loss_type, any_match=False, min_conf=0.25)

    self._cls_normalizer = self._cls_normalizer * self._classes / 80
    return

  def _compute_loss(self, true_counts, inds, y_true, boxes, classes, y_pred):
    """Per FPN path loss logic for Yolov4-csp, Yolov4-Large, and Yolov5."""
    # Generate shape constants.
    shape = tf.shape(true_counts)
    batch_size, width, height, num = shape[0], shape[1], shape[2], shape[3]
    fwidth = tf.cast(width, tf.float32)
    fheight = tf.cast(height, tf.float32)

    # Cast all input compontnts to float32 and stop gradient to save memory.
    y_true = tf.cast(y_true, tf.float32)
    true_counts = tf.cast(true_counts, tf.float32)
    true_conf = tf.clip_by_value(true_counts, 0.0, 1.0)
    grid_points, anchor_grid = self._anchor_generator(
        width, height, batch_size, dtype=tf.float32)

    # Split the y_true list.
    (true_box, ind_mask, true_class) = tf.split(y_true, [4, 1, 1], axis=-1)
    grid_mask = true_conf = tf.squeeze(true_conf, axis=-1)
    true_class = tf.squeeze(true_class, axis=-1)
    num_objs = tf.cast(tf.reduce_sum(ind_mask), dtype=y_pred.dtype)

    # Split up the predicitons.
    y_pred = tf.cast(
        tf.reshape(y_pred, [batch_size, width, height, num, -1]), tf.float32)
    pred_box, pred_conf, pred_class = tf.split(y_pred, [4, 1, -1], axis=-1)

    # Decode the boxes for loss compute.
    scale, pred_box, pbg = self._decode_boxes(
        fwidth, fheight, pred_box, anchor_grid, grid_points, darknet=False)

    # If the ignore threshold is enabled, search all boxes ignore all
    # IOU valeus larger than the ignore threshold that are not in the
    # noted ground truth list.
    if self._ignore_thresh != 0.0:
      (_, obj_mask) = self._tiled_global_box_search(
          pbg,
          tf.stop_gradient(tf.sigmoid(pred_class)),
          boxes,
          classes,
          true_conf,
          smoothed=False,
          scale=None)

    # Scale and shift and select the ground truth boxes
    # and predictions to the prediciton domain.
    if self._box_type == 'anchor_free':
      true_box = loss_utils.apply_mask(ind_mask,
                                       (scale * self._path_stride * true_box))
    else:
      offset = tf.cast(
          tf.gather_nd(grid_points, inds, batch_dims=1), true_box.dtype)
      offset = tf.concat([offset, tf.zeros_like(offset)], axis=-1)
      true_box = loss_utils.apply_mask(ind_mask, (scale * true_box) - offset)
    pred_box = loss_utils.apply_mask(ind_mask,
                                     tf.gather_nd(pred_box, inds, batch_dims=1))

    # Select the correct/used prediction classes.
    true_class = tf.one_hot(
        tf.cast(true_class, tf.int32),
        depth=tf.shape(pred_class)[-1],
        dtype=pred_class.dtype)
    true_class = loss_utils.apply_mask(ind_mask, true_class)
    pred_class = loss_utils.apply_mask(
        ind_mask, tf.gather_nd(pred_class, inds, batch_dims=1))

    # Compute the box loss.
    _, iou, box_loss = self.box_loss(true_box, pred_box, darknet=False)
    box_loss = loss_utils.apply_mask(tf.squeeze(ind_mask, axis=-1), box_loss)
    box_loss = math_ops.divide_no_nan(tf.reduce_sum(box_loss), num_objs)

    # Use the box IOU to build the map for confidence loss computation.
    iou = tf.maximum(tf.stop_gradient(iou), 0.0)
    smoothed_iou = ((
        (1 - self._objectness_smooth) * tf.cast(ind_mask, iou.dtype)) +
                    self._objectness_smooth * tf.expand_dims(iou, axis=-1))
    smoothed_iou = loss_utils.apply_mask(ind_mask, smoothed_iou)
    true_conf = loss_utils.build_grid(
        inds, smoothed_iou, pred_conf, ind_mask, update=self._update_on_repeat)
    true_conf = tf.squeeze(true_conf, axis=-1)

    # Compute the cross entropy loss for the confidence map.
    bce = tf.keras.losses.binary_crossentropy(
        tf.expand_dims(true_conf, axis=-1), pred_conf, from_logits=True)
    if self._ignore_thresh != 0.0:
      bce = loss_utils.apply_mask(obj_mask, bce)
      conf_loss = tf.reduce_sum(bce) / tf.reduce_sum(obj_mask)
    else:
      conf_loss = tf.reduce_mean(bce)

    # Compute the cross entropy loss for the class maps.
    class_loss = tf.keras.losses.binary_crossentropy(
        true_class,
        pred_class,
        label_smoothing=self._label_smoothing,
        from_logits=True)
    class_loss = loss_utils.apply_mask(
        tf.squeeze(ind_mask, axis=-1), class_loss)
    class_loss = math_ops.divide_no_nan(tf.reduce_sum(class_loss), num_objs)

    # Apply the weights to each loss.
    box_loss *= self._iou_normalizer
    class_loss *= self._cls_normalizer
    conf_loss *= self._object_normalizer

    # Add all the losses together then take the sum over the batches.
    mean_loss = box_loss + class_loss + conf_loss
    loss = mean_loss * tf.cast(batch_size, mean_loss.dtype)

    return (loss, box_loss, conf_loss, class_loss, mean_loss, iou, pred_conf,
            ind_mask, grid_mask)

  def post_path_aggregation(self, loss, box_loss, conf_loss, class_loss,
                            ground_truths, predictions):
    """This method allows for post processing of a loss value.

    By default the model will have about 3 FPN levels {3, 4, 5}, on
    larger model that have more like 4 or 5 FPN levels the loss needs to
    be scaled such that the total update is scaled to the same effective
    magintude as the model with 3 FPN levels. This helps to prevent gradient
    explosions.

    Args:
      loss: `tf.float` scalar for the actual loss.
      box_loss: `tf.float` for the loss on the boxs only.
      conf_loss: `tf.float` for the loss on the confidences only.
      class_loss: `tf.float` for the loss on the classes only.
      ground_truths: `Dict` holding all the ground truth tensors.
      predictions: `Dict` holding all the predicted values.
    Returns:
      loss: `tf.float` scalar for the scaled loss.
      scale: `tf.float` how much the loss was scaled by.
    """
    scale = tf.stop_gradient(3 / len(list(predictions.keys())))
    return loss * scale, 1 / scale

  def cross_replica_aggregation(self, loss, num_replicas_in_sync):
    """This method is not specific to each loss path, but each loss type."""
    return loss


class YoloLoss:
  """This class implements the aggregated loss across YOLO model FPN levels."""

  def __init__(self,
               keys,
               classes,
               anchors,
               path_strides=None,
               truth_thresholds=None,
               ignore_thresholds=None,
               loss_types=None,
               iou_normalizers=None,
               cls_normalizers=None,
               object_normalizers=None,
               objectness_smooths=None,
               box_types=None,
               scale_xys=None,
               max_deltas=None,
               label_smoothing=0.0,
               use_scaled_loss=False,
               update_on_repeat=True):
    """Loss Function Initialization.

    Args:
      keys: `List[str]` indicating the name of the FPN paths that need to be
        optimized.
      classes: `int` for the number of classes
      anchors: `List[List[int]]` for the anchor boxes that are used in the model
        at all levels. For anchor free prediction set the anchor list to be the
        same as the image resolution.
      path_strides: `Dict[int]` for how much to scale this level to get the
        orginal input shape for each FPN path.
      truth_thresholds: `Dict[float]` for the IOU value over which the loss is
        propagated despite a detection being made for each FPN path.
      ignore_thresholds: `Dict[float]` for the IOU value over which the loss is
        not propagated, and a detection is assumed to have been made for each
        FPN path.
      loss_types: `Dict[str]` for the typeof iou loss to use with in {ciou,
        diou, giou, iou} for each FPN path.
      iou_normalizers: `Dict[float]` for how much to scale the loss on the IOU
        or the boxes for each FPN path.
      cls_normalizers: `Dict[float]` for how much to scale the loss on the
        classes for each FPN path.
      object_normalizers: `Dict[float]` for how much to scale loss on the
        detection map for each FPN path.
      objectness_smooths: `Dict[float]` for how much to smooth the loss on the
        detection map for each FPN path.
      box_types: `Dict[bool]` for which scaling type to use for each FPN path.
      scale_xys:  `Dict[float]` values inidcating how far each pixel can see
        outside of its containment of 1.0. a value of 1.2 indicates there is a
        20% extended radius around each pixel that this specific pixel can
        predict values for a center at. the center can range from 0 - value/2 to
        1 + value/2, this value is set in the yolo filter, and resused here.
        there should be one value for scale_xy for each level from min_level to
        max_level. One for each FPN path.
      max_deltas: `Dict[float]` for gradient clipping to apply to the box loss
        for each FPN path.
      label_smoothing: `Dict[float]` for how much to smooth the loss on the
        classes for each FPN path.
      use_scaled_loss: `bool` for whether to use the scaled loss or the
        traditional loss.
      update_on_repeat: `bool` for whether to replace with the newest or the
        best value when an index is consumed by multiple objects.
    """

    losses = {'darknet': DarknetLoss, 'scaled': ScaledLoss}

    if use_scaled_loss:
      loss_type = 'scaled'
    else:
      loss_type = 'darknet'

    self._loss_dict = {}
    for key in keys:
      self._loss_dict[key] = losses[loss_type](
          classes=classes,
          anchors=anchors[key],
          truth_thresh=truth_thresholds[key],
          ignore_thresh=ignore_thresholds[key],
          loss_type=loss_types[key],
          iou_normalizer=iou_normalizers[key],
          cls_normalizer=cls_normalizers[key],
          object_normalizer=object_normalizers[key],
          box_type=box_types[key],
          objectness_smooth=objectness_smooths[key],
          max_delta=max_deltas[key],
          path_stride=path_strides[key],
          scale_x_y=scale_xys[key],
          update_on_repeat=update_on_repeat,
          label_smoothing=label_smoothing)

  def __call__(self, ground_truth, predictions):
    metric_dict = collections.defaultdict(dict)
    metric_dict['net']['box'] = 0
    metric_dict['net']['class'] = 0
    metric_dict['net']['conf'] = 0

    loss_val, metric_loss = 0, 0
    num_replicas_in_sync = tf.distribute.get_strategy().num_replicas_in_sync

    for key in predictions.keys():
      (loss, loss_box, loss_conf, loss_class, mean_loss, avg_iou,
       avg_obj) = self._loss_dict[key](ground_truth['true_conf'][key],
                                       ground_truth['inds'][key],
                                       ground_truth['upds'][key],
                                       ground_truth['bbox'],
                                       ground_truth['classes'],
                                       predictions[key])

      # after computing the loss, scale loss as needed for aggregation
      # across FPN levels
      loss, scale = self._loss_dict[key].post_path_aggregation(
          loss, loss_box, loss_conf, loss_class, ground_truth, predictions)

      # after completing the scaling of the loss on each replica, handle
      # scaling the loss for mergeing the loss across replicas
      loss = self._loss_dict[key].cross_replica_aggregation(
          loss, num_replicas_in_sync)
      loss_val += loss

      # detach all the below gradients: none of them should make a
      # contribution to the gradient form this point forwards
      metric_loss += tf.stop_gradient(mean_loss / scale)
      metric_dict[key]['loss'] = tf.stop_gradient(mean_loss / scale)
      metric_dict[key]['avg_iou'] = tf.stop_gradient(avg_iou)
      metric_dict[key]['avg_obj'] = tf.stop_gradient(avg_obj)

      metric_dict['net']['box'] += tf.stop_gradient(loss_box / scale)
      metric_dict['net']['class'] += tf.stop_gradient(loss_class / scale)
      metric_dict['net']['conf'] += tf.stop_gradient(loss_conf / scale)

    return loss_val, metric_loss, metric_dict

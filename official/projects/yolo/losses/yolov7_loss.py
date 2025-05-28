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

"""YOLOv7 loss function."""

import tensorflow as tf, tf_keras

from official.projects.yolo.ops import box_ops
from official.vision.losses import focal_loss

_LAYER_BALANCE = {
    '3': [4.0, 1.0, 0.4],
    '5': [4.0, 1.0, 0.25, 0.06, 0.02],
}


def smooth_bce_targets(eps=0.1):
  """Computes positive, negative label smoothing BCE targets.

  https://arxiv.org/pdf/1902.04103.pdf equation 3.

  Args:
    eps: a float number from [0, 1] representing label smoothing factor.

  Returns:
    Positive and negative targets after label smoothing.
  """
  return 1.0 - 0.5 * eps, 0.5 * eps


def merge_labels(labels):
  """Converts the ground-truth labels into loss targets."""
  boxes = box_ops.yxyx_to_xcycwh(labels['bbox'])
  classes = tf.cast(labels['classes'], boxes.dtype)
  return tf.concat([classes[..., None], boxes], axis=-1)


class YoloV7Loss(tf_keras.losses.Loss):
  """YOLOv7 loss function."""

  def __init__(
      self,
      anchors,
      strides,
      input_size,
      alpha=0.25,
      gamma=1.5,
      box_weight=0.05,
      obj_weight=0.7,
      cls_weight=0.3,
      label_smoothing=0.0,
      anchor_threshold=4.0,
      iou_mix_ratio=1.0,
      num_classes=80,
      auto_balance=False,
      reduction=tf_keras.losses.Reduction.NONE,
      name=None,
  ):
    """Constructor for YOLOv7 loss.

    Follows the implementation here:
      https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py#L422

    Args:
      anchors: a 2D array represents different anchors used at each level.
      strides: a 1D array represents the strides. Note that all numbers should
        be a power of 2, and they usually start with level 3 and end at level
        5 or 7. Therefore, the list should usually be [8, 16, 32] or
        [8, 16, 32, 64, 128].
      input_size: a list containing the height and width of the input image.
      alpha: alpha for focal loss.
      gamma: gamma for focal loss. If set to 0, focal loss will be disabled.
      box_weight: float weight scalar applied to bounding box loss.
      obj_weight: float weight scalar applied to objectness loss.
      cls_weight: float weight scalar applied to class loss.
      label_smoothing: small float number used to compute positive and negative
        targets. If set to 0, the positive targets will be 1 and negative
        targets will be 0.
      anchor_threshold: threshold for the anchor matching. Larger number allows
        more displacements between anchors and targets.
      iou_mix_ratio: float ratio to mix the IoU score with the positive target,
        which is 1.
      num_classes: number of classes.
      auto_balance: a boolean flag that indicates whether auto balance should be
        used. If used, the default balance factors will automatically update
        for each batch.
      reduction: Reduction method. Should be set to None at all time as this
        loss module always output a loss scalar.
      name: Optional name for the loss.
    """
    # Loss required fields.
    self._num_classes = num_classes
    self._num_layers = len(strides)
    self._num_anchors = len(anchors[0])
    self._anchors = anchors
    self._strides = strides
    self._input_size = input_size
    self._iou_mix_ratio = iou_mix_ratio

    # Scale down anchors by the strides to match the feature map.
    for i, stride in enumerate(strides):
      self._anchors[i] = tf.constant(self._anchors[i], tf.float32) / stride

    self._anchor_threshold = anchor_threshold

    self._pos_targets, self._neg_targets = smooth_bce_targets(label_smoothing)
    if gamma > 0:
      self._cls_loss_fn = focal_loss.FocalLoss(
          alpha=alpha, gamma=gamma, reduction=reduction, name='cls_loss')
      self._obj_loss_fn = focal_loss.FocalLoss(
          alpha=alpha, gamma=gamma, reduction=reduction, name='obj_loss')
    else:
      self._cls_loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
      self._obj_loss_fn = tf.nn.sigmoid_cross_entropy_with_logits

    # Weight to combine losses
    self._box_weight = box_weight
    self._obj_weight = obj_weight * input_size[0] / 640 * input_size[1] / 640
    self._cls_weight = cls_weight * num_classes / 80

    # Layer balance scalar
    self._balance = _LAYER_BALANCE[str(self._num_layers)][:]
    for i, bal in enumerate(self._balance):
      self._balance[i] = tf.constant(bal, tf.float32)
    self._auto_balance = auto_balance
    assert 16 in strides, (
        'Expect level 4 (stride of 16) always exist in the strides, received %s'
        % strides
    )
    self._ssi = list(strides).index(16) if auto_balance else 0  # stride 16 idx

    super().__init__(reduction=reduction, name=name)

  def call(self, labels, predictions):
    labels = merge_labels(labels)
    p = {}
    for key in predictions:
      # [batch_size, num_anchors, height, width, num_classes + boxes + obj]
      p[key] = tf.transpose(predictions[key], [0, 3, 1, 2, 4])
    cls_loss, box_loss, obj_loss, iou_metric = [tf.zeros(1) for _ in range(4)]
    total_num_matchings = tf.zeros(1)
    total_num_gts = tf.reduce_sum(tf.cast(labels[..., 0] != -1, tf.float32))

    masks, indices, anchors, cls_targets, box_targets = self._build_targets(
        labels, p)

    batch_size = tf.shape(indices)[0]
    layer_shape = [batch_size, self._num_layers, -1]
    # [anchor_indices, grid_js, grid_is]
    masks = tf.reshape(masks, layer_shape)
    indices = tf.reshape(indices, [*layer_shape, 3])
    anchors = tf.reshape(anchors, [*layer_shape, 2])
    cls_targets = tf.reshape(cls_targets, layer_shape)
    box_targets = tf.reshape(box_targets, [*layer_shape, 4])

    # Losses
    for layer_key, layer_pred in p.items():
      i = int(layer_key) - 3

      obj_targets = tf.zeros_like(layer_pred[..., 0])

      layer_masks = masks[:, i]
      num_matchings = tf.reduce_sum(tf.cast(layer_masks, tf.int32))
      total_num_matchings += tf.cast(num_matchings, tf.float32)

      if num_matchings > 0:
        layer_indices = indices[:, i]
        batch_indices = tf.tile(
            tf.range(batch_size)[:, None], [1, tf.shape(layer_indices)[1]]
        )[..., None]
        layer_indices = tf.concat([batch_indices, layer_indices], axis=-1)
        layer_indices = tf.boolean_mask(layer_indices, layer_masks)
        layer_anchors = tf.boolean_mask(anchors[:, i], layer_masks)

        layer_cls_targets = tf.boolean_mask(cls_targets[:, i], layer_masks)
        layer_box_targets = tf.boolean_mask(box_targets[:, i], layer_masks)

        # In the same shape of layer_target.
        matched_pred = tf.gather_nd(layer_pred, layer_indices)

        pred_xcyc = tf.sigmoid(matched_pred[..., :2]) * 2 - 0.5
        pred_wh = (
            tf.square(tf.sigmoid(matched_pred[..., 2:4]) * 2) * layer_anchors)
        pred_xcycwh = tf.concat([pred_xcyc, pred_wh], axis=-1)
        _, ciou = box_ops.compute_ciou(pred_xcycwh, layer_box_targets)

        box_loss += tf.reduce_mean(1.0 - ciou)
        iou_metric += tf.reduce_mean(ciou)

        # Compute classification loss.
        if self._num_classes > 1:  # cls loss (only if multiple classes)
          t = tf.one_hot(
              layer_cls_targets,
              self._num_classes,
              on_value=self._pos_targets,
              off_value=self._neg_targets,
          )
          cls_loss += tf.reduce_mean(
              self._cls_loss_fn(t, matched_pred[..., 5:]))

        # Compute objectness loss.
        iou_ratio = tf.cast(
            (1.0 - self._iou_mix_ratio)
            + (self._iou_mix_ratio * tf.maximum(tf.stop_gradient(ciou), 0)),
            obj_targets.dtype,
        )
        obj_targets = tf.tensor_scatter_nd_max(
            obj_targets, layer_indices, iou_ratio
        )
      layer_obj_loss = tf.reduce_mean(
          self._obj_loss_fn(obj_targets, layer_pred[..., 4])
      )
      obj_loss += layer_obj_loss * self._balance[i]
      # Updates the balance factor, which is a moving average of previous
      # factor at the same level.
      if self._auto_balance:
        self._balance[i] = self._balance[
            i
        ] * 0.9999 + 0.0001 / tf.stop_gradient(layer_obj_loss)

    # Re-balance the factors so that stride at self._ssi always receives 1.
    if self._auto_balance:
      self._balance = [x / self._balance[self._ssi] for x in self._balance]

    box_loss *= self._box_weight
    obj_loss *= self._obj_weight
    cls_loss *= self._cls_weight

    self._box_loss = tf.stop_gradient(box_loss)
    self._obj_loss = tf.stop_gradient(obj_loss)
    self._cls_loss = tf.stop_gradient(cls_loss)
    self._iou = tf.stop_gradient(iou_metric) / self._num_layers
    self._num_matchings = tf.stop_gradient(
        total_num_matchings) / tf.cast(batch_size, tf.float32)
    self._num_gts = tf.stop_gradient(
        total_num_gts) / tf.cast(batch_size, tf.float32)

    loss = box_loss + obj_loss + cls_loss
    return loss * tf.cast(batch_size, loss.dtype)

  def _build_targets(self, labels, predictions):
    """Finds three matching anchors for each ground-truth."""
    label_shape = tf.shape(labels)
    batch_size, max_boxes = label_shape[0], label_shape[1]
    masks, indices, anch = [], [], []
    cls_targets, box_targets = [], []
    anchor_indices = tf.tile(
        tf.range(self._num_anchors, dtype=tf.float32)[None, None],
        [batch_size, max_boxes, 1],
    )
    # Append anchor indices to labels.
    labels = tf.tile(labels[:, :, None], [1, 1, self._num_anchors, 1])
    labels = tf.concat([labels, anchor_indices[..., None]], axis=-1)

    # Bias is used to determine the matching. 0.5 means matching anchors that
    # fall in the 0.5 differences in the feature map. For instance, a box
    # coordinates of (15.6, 35.4) will match the anchors at [15, 35], [16, 35],
    # and [15, 34].
    bias = 0.5  # bias
    off = (
        tf.constant(
            [
                [0, 0],
                [1, 0], [0, 1], [-1, 0], [0, -1],  # j, k, l, m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            tf.float32,
        )
        * bias
    )  # offsets

    for i in range(self._num_layers):
      anchors = self._anchors[i]
      _, _, h, w, _ = predictions[str(i + 3)].get_shape().as_list()
      gain = tf.constant([1, w, h, w, h, 1], dtype=tf.float32)

      t = labels * gain

      # Filter out targets that do not match the current anchors.
      wh_ratio = t[..., 3:5] / tf.cast(anchors[None, None], tf.float32)
      labels_mask = tf.less(
          tf.reduce_max(tf.maximum(wh_ratio, 1.0 / wh_ratio), axis=-1),
          self._anchor_threshold,
      )[..., None]
      # Compute valid mask for ground-truths.
      labels_mask = tf.logical_and(t[..., :1] != -1, labels_mask)

      labels_mask = tf.reshape(labels_mask, [batch_size, -1])
      t = tf.reshape(t, [batch_size, -1, 6])

      # Find the matching offsets for valid labels.
      gxy = t[..., 1:3]  # grid xy
      gxi = gain[1:3] - gxy  # inverse
      j, k = tf.split((gxy % 1.0 < bias) & (gxy >= 1.0), 2, axis=-1)
      l, m = tf.split((gxi % 1.0 < bias) & (gxi >= 1.0), 2, axis=-1)

      j, k, l, m = j[..., 0], k[..., 0], l[..., 0], m[..., 0]

      # Note that j and l, k and m are conjugate to each other, so at most one
      # of them will be True during running. Therefore, we can reduce memory
      # usage by gathering the selected index.
      x_map = tf.cast(tf.stack([j, l], axis=-1), tf.int8)
      y_map = tf.cast(tf.stack([k, m], axis=-1), tf.int8)

      # Add the indices offsets.
      x_indices = tf.argmax(x_map, axis=-1) * 2 + 1
      y_indices = tf.argmax(y_map, axis=-1) * 2 + 2
      three_targets_indices = tf.stack(
          [tf.zeros_like(x_indices), x_indices, y_indices], axis=-1
      )[..., None]

      # Gather the selected 3 targets from the 5-target map.
      j = tf.stack([tf.ones_like(j), j, k, l, m], axis=-1)
      three_targets_mask = tf.gather_nd(j, three_targets_indices, batch_dims=2)

      labels_mask = tf.tile(labels_mask[:, :, None], [1, 1, 5])
      t = tf.tile(t[:, :, None], [1, 1, 5, 1])

      labels_mask = tf.gather_nd(
          labels_mask, three_targets_indices, batch_dims=2
      )
      t = tf.gather_nd(t, three_targets_indices, batch_dims=2)

      offsets = tf.zeros_like(gxy)[:, :, None] + off[None, None]
      offsets = tf.gather_nd(offsets, three_targets_indices, batch_dims=2)

      cls_target = tf.cast(t[..., 0], tf.int32)
      gxy, gwh = t[..., 1:3], t[..., 3:5]
      # Find the actual grid locations.
      gij = tf.cast(gxy - offsets * 2, tf.int32)
      gi, gj = tf.split(gij, 2, axis=-1)
      gi, gj = gi[..., 0], gj[..., 0]

      # Append the result.
      anchor_idx = tf.cast(t[..., 5], tf.int32)
      gain = tf.cast(gain, tf.int32)
      gi = tf.clip_by_value(gi, 0, gain[2] - 1)
      gj = tf.clip_by_value(gj, 0, gain[3] - 1)
      gij = tf.stack([gi, gj], axis=-1)

      labels_mask = tf.logical_and(labels_mask, three_targets_mask)
      masks.append(labels_mask)
      indices.append(tf.stack([anchor_idx, gj, gi], axis=-1))
      anch.append(tf.gather(anchors, anchor_idx))
      cls_targets.append(cls_target)
      box_targets.append(
          tf.concat([gxy - tf.cast(gij, tf.float32), gwh], axis=-1))  # box

    # [batch_size, num_layers, num_anchors * max_boxes, num_targets]
    masks = tf.stack(masks, axis=1)
    indices = tf.stack(indices, axis=1)
    anch = tf.stack(anch, axis=1)
    cls_targets = tf.stack(cls_targets, axis=1)
    box_targets = tf.stack(box_targets, axis=1)
    return masks, indices, anch, cls_targets, box_targets

  def report_separate_losses(self):
    return {
        'box_loss': self._box_loss,
        'obj_loss': self._obj_loss,
        'cls_loss': self._cls_loss,
        'iou': self._iou,
    }

  def report_stats(self):
    return {
        'num_gts': self._num_gts,
        'num_matchings': self._num_matchings,
        # No duplicates.
        'num_duplicates': tf.constant(0),
    }

  def get_config(self):
    config = {
        'alpha': self._alpha,
        'gamma': self._gamma,
        'box_weight': self._box_weight,
        'obj_weight': self._obj_weight,
        'cls_weight': self._cls_weight,
        'pos_targets': self._pos_targets,
        'neg_targets': self._neg_targets,
        'num_classes': self._num_classes,
        'num_layers': self._num_layers,
        'num_anchors': self._num_anchors,
        'auto_balance': self._auto_balance,
        'balance': self._balance,
        'strides': self._strides,
        'anchors': self._anchors,
        'input_size': self._input_size,
        'anchor_threshold': self._anchor_threshold,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class YoloV7LossOTA(tf_keras.losses.Loss):
  """YOLOv7 loss function with OTA.

  OTA (Optimal Transport Assignment) uses Sinkhorn-Knopp algorithm to copmute
  a matching between anchors and ground-truth labels.

  Paper: https://arxiv.org/pdf/2103.14259.pdf
  """

  def __init__(
      self,
      anchors,
      strides,
      input_size,
      alpha=0.25,
      gamma=1.5,
      box_weight=0.05,
      obj_weight=0.7,
      cls_weight=0.3,
      iou_weight=3.0,
      label_smoothing=0.0,
      anchor_threshold=4.0,
      iou_mix_ratio=1.0,
      num_classes=80,
      auto_balance=False,
      reduction=tf_keras.losses.Reduction.NONE,
      name=None,
  ):
    """Constructor for YOLOv7 loss OTA.

    Follows the implementation here:
      https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py#L556

    Args:
      anchors: a 2D array represents different anchors used at each level.
      strides: a 1D array represents the strides. Note that all numbers should
        be a power of 2, and they usually start with level 3 and end at level 5
        or 7. Therefore, the list should usually be [8, 16, 32] or [8, 16, 32,
        64, 128].
      input_size: a list containing the height and width of the input image.
      alpha: alpha for focal loss.
      gamma: gamma for focal loss. If set to 0, focal loss will be disabled.
      box_weight: float weight scalar applied to bounding box loss.
      obj_weight: float weight scalar applied to objectness loss.
      cls_weight: float weight scalar applied to class loss.
      iou_weight: float weight scalar to mix class loss and IoU class to
        construct the cost matrix.
      label_smoothing: small float number used to compute positive and negative
        targets. If set to 0, the positive targets will be 1 and negative
        targets will be 0.
      anchor_threshold: threshold for the anchor matching. Larger number allows
        more displacements between anchors and targets.
      iou_mix_ratio: float ratio to mix the IoU score with the positive target,
        which is 1.
      num_classes: number of classes.
      auto_balance: a boolean flag that indicates whether auto balance should be
        used. If used, the default balance factors will automatically update for
        each batch.
      reduction: Reduction method. Should be set to None at all time as this
        loss module always output a loss scalar.
      name: Optional name for the loss.
    """
    # Loss required fields.
    self._num_classes = num_classes
    self._num_layers = len(strides)
    self._num_anchors = len(anchors[0])
    self._anchors = []
    self._strides = strides
    self._input_size = input_size
    self._iou_mix_ratio = iou_mix_ratio

    # Scale down anchors by the strides to match the feature map.
    for i, stride in enumerate(strides):
      self._anchors.append(tf.constant(anchors[i], tf.float32) / stride)

    self._anchor_threshold = anchor_threshold

    self._pos_targets, self._neg_targets = smooth_bce_targets(label_smoothing)
    if gamma > 0:
      self._cls_loss_fn = focal_loss.FocalLoss(
          alpha=alpha, gamma=gamma, reduction=reduction, name='cls_loss')
      self._obj_loss_fn = focal_loss.FocalLoss(
          alpha=alpha, gamma=gamma, reduction=reduction, name='obj_loss')
    else:
      self._cls_loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
      self._obj_loss_fn = tf.nn.sigmoid_cross_entropy_with_logits

    # Weight to combine losses
    self._box_weight = box_weight
    self._obj_weight = obj_weight * input_size[0] / 640 * input_size[1] / 640
    self._cls_weight = cls_weight * num_classes / 80

    # Weight to construct cost matrix
    self._iou_weight = iou_weight

    # Layer balance scalar
    self._balance = _LAYER_BALANCE[str(self._num_layers)][:]
    for i, bal in enumerate(self._balance):
      self._balance[i] = tf.constant(bal, tf.float32)
    self._auto_balance = auto_balance
    assert 16 in strides, (
        'Expect level 4 (stride of 16) always exist in the strides, received %s'
        % strides
    )
    self._ssi = list(strides).index(16) if auto_balance else 0  # stride 16 idx

    super().__init__(reduction=reduction, name=name)

  def call(self, labels, predictions):
    """Comptues the OTA loss.

    Args:
      labels: a dictionary contains the following required keys:
        - classes: class indices in shape [batch_size, max_num_instances].
        - bbox: bounding boxes in shape [batch_size, max_num_instances, 4].
        - image_info: image info in shape [batch_size, 4, 2].
      predictions: a dictionary contains model outputs at different layers.
        They are in shape of [batch_size, h_at_level, w_at_level, num_anchors,
        num_classes + 4 (box coordinates) + 1 (objectness)].

    Returns:
      The scaled loss (up by batch size) from OTA.
    """
    image_info = labels['image_info']
    # Convert labels dictionary into tensors.
    labels = merge_labels(labels)
    p = {}
    for key in predictions:
      # [batch_size, num_anchors, height, width, num_classes + boxes + obj]
      p[key] = tf.transpose(predictions[key], [0, 3, 1, 2, 4])

    cls_loss, box_loss, obj_loss, iou_metric = [tf.zeros(1) for _ in range(4)]
    total_num_matchings = tf.zeros(1)
    total_num_gts = tf.reduce_sum(tf.cast(labels[..., 0] != -1, tf.float32))
    (matched_indices, matched_anchors, matched_mask, matched_targets,
     num_duplicates) = self._build_targets(labels, p, image_info)
    # Get height and width for each layers.
    pre_gen_gains = [
        tf.gather(tf.shape(p[str(i + 3)]), [3, 2, 3, 2])
        for i in range(self._num_layers)
    ]

    batch_size = tf.shape(matched_indices)[0]
    layer_shape = [batch_size, self._num_layers, -1]
    # [anchor_indices, grid_js, grid_is]
    masks = tf.reshape(matched_mask, layer_shape)
    indices = tf.reshape(matched_indices, [*layer_shape, 3])
    anchors = tf.reshape(matched_anchors, [*layer_shape, 2])
    targets = tf.reshape(matched_targets, [*layer_shape, 5])

    # Losses
    for layer_idx, layer_pred in p.items():
      # Always assume the output level starts with 3.
      i = int(layer_idx) - 3

      obj_targets = tf.zeros_like(layer_pred[..., 0])

      # Get layer inputs
      layer_masks = masks[:, i]
      num_matchings = tf.reduce_sum(tf.cast(layer_masks, tf.int32))
      total_num_matchings += tf.cast(num_matchings, tf.float32)

      if num_matchings > 0:
        layer_indices = indices[:, i]
        batch_indices = tf.tile(
            tf.range(batch_size)[:, None], [1, tf.shape(layer_indices)[1]]
        )[..., None]
        layer_indices = tf.concat([batch_indices, layer_indices], axis=-1)
        layer_indices = tf.boolean_mask(layer_indices, layer_masks)
        layer_anchors = tf.boolean_mask(anchors[:, i], layer_masks)

        layer_targets = tf.boolean_mask(targets[:, i], layer_masks)
        layer_cls_targets = tf.cast(layer_targets[:, 0], tf.int32)
        layer_box_targets = layer_targets[:, 1:]

        # In the same shape of layer_target.
        matched_pred = tf.gather_nd(layer_pred, layer_indices)

        pred_xcyc = tf.sigmoid(matched_pred[..., :2]) * 2 - 0.5
        pred_wh = (
            tf.square(tf.sigmoid(matched_pred[..., 2:4]) * 2) * layer_anchors)
        pred_xcycwh = tf.concat([pred_xcyc, pred_wh], axis=-1)

        grid = tf.cast(
            tf.stack(
                [
                    layer_indices[:, 3],  # gi
                    layer_indices[:, 2],  # gj
                    tf.zeros_like(layer_indices[:, 0]),
                    tf.zeros_like(layer_indices[:, 0]),
                ],
                axis=-1,
            ),
            tf.float32,
        )
        target_xcycwh = layer_box_targets * tf.cast(
            pre_gen_gains[i], layer_targets.dtype
        )
        target_xcycwh -= grid
        _, ciou = box_ops.compute_ciou(target_xcycwh, pred_xcycwh)

        box_loss += tf.reduce_mean(1.0 - ciou)
        iou_metric += tf.reduce_mean(ciou)

        # Compute classification loss.
        if self._num_classes > 1:  # cls loss (only if multiple classes)
          t = tf.one_hot(
              layer_cls_targets,
              self._num_classes,
              on_value=self._pos_targets,
              off_value=self._neg_targets,
          )
          cls_loss += tf.reduce_mean(
              self._cls_loss_fn(t, matched_pred[..., 5:]))

        # Compute objectness loss.
        iou_ratio = tf.cast(
            (1.0 - self._iou_mix_ratio)
            + (self._iou_mix_ratio * tf.maximum(tf.stop_gradient(ciou), 0)),
            obj_targets.dtype,
        )
        obj_targets = tf.tensor_scatter_nd_max(
            obj_targets, layer_indices, iou_ratio
        )
      layer_obj_loss = tf.reduce_mean(
          self._obj_loss_fn(obj_targets, layer_pred[..., 4])
      )
      obj_loss += layer_obj_loss * self._balance[i]
      # Updates the balance factor, which is a moving average of previous
      # factor at the same level.
      if self._auto_balance:
        self._balance[i] = self._balance[
            i
        ] * 0.9999 + 0.0001 / tf.stop_gradient(layer_obj_loss)

    # Re-balance the factors so that stride at self._ssi always receives 1.
    if self._auto_balance:
      self._balance = [x / self._balance[self._ssi] for x in self._balance]

    # Keep separate losses for summary purpose.
    box_loss *= self._box_weight
    obj_loss *= self._obj_weight
    cls_loss *= self._cls_weight

    self._iou = tf.stop_gradient(iou_metric) / self._num_layers
    self._num_matchings = tf.stop_gradient(
        total_num_matchings) / tf.cast(batch_size, tf.float32)
    self._num_gts = total_num_gts / tf.cast(batch_size, tf.float32)
    self._num_duplicates = tf.stop_gradient(
        num_duplicates) / tf.cast(batch_size, tf.float32)
    self._box_loss = tf.stop_gradient(box_loss)
    self._obj_loss = tf.stop_gradient(obj_loss)
    self._cls_loss = tf.stop_gradient(cls_loss)

    loss = box_loss + obj_loss + cls_loss

    # Scale up the loss by batch size.
    return loss * tf.cast(batch_size, loss.dtype)

  def _build_targets(self, labels, predictions, image_info):
    """Finds the matching targets using Sinkhorn-Knopp."""
    # Find the three positives matching first for predictions.
    masks, indices, anchors = self._find_three_positives(labels, predictions)

    batch_size = tf.shape(masks)[0]

    # Collect the predictions.
    p_box, p_cls, p_obj = [], [], []
    for layer_key, layer_p in predictions.items():
      # Always assume level starts from 3.
      i = int(layer_key) - 3
      layer_indices = tf.reshape(indices[:, i], [batch_size, -1, 3])
      anchor = tf.reshape(anchors[:, i], [batch_size, -1, 2])

      fg_pred = tf.gather_nd(layer_p, layer_indices, batch_dims=1)

      grid = tf.stack([layer_indices[..., 2], layer_indices[..., 1]], axis=-1)
      grid = tf.cast(grid, fg_pred.dtype)

      pxy = (tf.sigmoid(fg_pred[..., :2]) * 2 - 0.5 + grid) * self._strides[i]
      pwh = (
          tf.square(tf.sigmoid(fg_pred[..., 2:4]) * 2)
          * anchor
          * self._strides[i]
      )
      pxywh = tf.concat([pxy, pwh], axis=-1)

      p_box.append(pxywh)
      p_obj.append(fg_pred[..., 4:5])
      p_cls.append(fg_pred[..., 5:])

    p_box = tf.concat(p_box, axis=1)
    p_cls = tf.concat(p_cls, axis=1)
    p_obj = tf.concat(p_obj, axis=1)

    # Compute valid masks for both targets and predictions.
    t_mask = labels[..., 0] != -1
    p_mask = tf.reshape(masks, [batch_size, -1])
    # [anchor_idx, gj, gi]
    indices = tf.reshape(indices, [batch_size, -1, 3])
    anchors = tf.reshape(anchors, [batch_size, -1, 2])

    num_preds = tf.shape(p_box)[1]
    num_gts = tf.shape(labels)[1]

    # Computes pair-wise IoU.
    t_box = labels[..., 1:5] * tf.tile(image_info[0, 1], [2])

    pair_wise_iou = box_ops.compute_iou(t_box[:, :, None], p_box[:, None])
    pair_wise_iou_loss = -tf.math.log(pair_wise_iou + 1e-8)

    # Computes pair-wise class loss.
    y = tf.sqrt(tf.sigmoid(p_cls) * tf.sigmoid(p_obj))
    # Add 1e-9 to avoid nan.
    logits = tf.math.log(y / (1 - y + 1e-9) + 1e-9)
    logits = tf.tile(logits[:, None], [1, num_gts, 1, 1])

    t_cls = tf.cast(labels[..., 0], tf.int32)
    class_labels = tf.one_hot(t_cls, self._num_classes, dtype=tf.float32)
    class_labels = tf.tile(class_labels[:, :, None], [1, 1, num_preds, 1])

    pair_wise_cls_loss = tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(class_labels, logits), axis=-1
    )

    # Compute the cost matrix and its corresponding valid mask.
    cost_mask = tf.logical_and(t_mask[..., None], p_mask[:, None])
    cost = tf.stop_gradient(pair_wise_cls_loss + 3 * pair_wise_iou_loss)
    largest_cost = tf.reduce_max(cost)

    # Set invalid IoU to 0.0 for top_k.
    valid_iou = tf.where(cost_mask, pair_wise_iou, tf.zeros_like(pair_wise_iou))

    # Compute top-10 IoUs from valid IoUs for each target.
    # When matched predictions is smaller than 10, we only want the top-k where
    # k is the total size of the matched predictions (k < 10).
    top_k_mask = tf.less(
        tf.range(10)[None],
        tf.minimum(10, tf.reduce_sum(tf.cast(p_mask, tf.int32), axis=-1))[
            :, None
        ],
    )
    top_k_mask = tf.logical_and(top_k_mask[:, None], t_mask[..., None])
    top_k, _ = tf.nn.top_k(valid_iou, k=10)
    top_k = tf.where(top_k_mask, top_k, tf.zeros_like(top_k))

    # Use top_k to compute the dynamic ks for target matching. Each target_i can
    # match to k_i predictions, and k_i is computed based on the pair-wise
    # valid IoU.
    dynamic_ks = tf.maximum(tf.cast(tf.reduce_sum(top_k, axis=-1), tf.int32), 1)
    dynamic_ks = tf.where(t_mask, dynamic_ks, tf.zeros_like(dynamic_ks))
    dynamic_ks = tf.stop_gradient(dynamic_ks)
    dynamic_mask = tf.range(10)[None, None] < dynamic_ks[..., None]

    # Set the invalid field to maximum cost so that they won't be selected
    # during matching.
    cost = tf.where(cost_mask, cost, tf.ones_like(cost) * (largest_cost + 1))

    matching_matrix = tf.zeros_like(cost, dtype=tf.int32)
    _, pred_idx = tf.nn.top_k(-cost, k=10)

    # Update matching matrix.
    # [batch_size, num_gts, 10]
    batch_idx = tf.tile(tf.range(batch_size)[:, None, None], [1, num_gts, 10])
    gt_idx = tf.tile(tf.range(num_gts)[None, :, None], [batch_size, 1, 10])
    matched_indices = tf.stack([batch_idx, gt_idx, pred_idx], axis=-1)
    matching_matrix = tf.tensor_scatter_nd_add(
        matching_matrix,
        matched_indices,
        tf.cast(dynamic_mask, matching_matrix.dtype),
    )

    # Detect if there is a detection matches to multiple targets, if so, we
    # assign it to the target with minimum cost.
    duplicate_mask = tf.reduce_sum(matching_matrix, axis=1) > 1
    num_duplicates = tf.reduce_sum(tf.cast(duplicate_mask, tf.float32))
    cost_argmin = tf.argmin(cost, axis=1, output_type=tf.int32)

    remove_mask = tf.tile(duplicate_mask[:, None], [1, num_gts, 1])
    matching_matrix = tf.where(
        remove_mask, tf.zeros_like(matching_matrix), matching_matrix)

    min_mask = tf.equal(
        tf.tile(tf.range(num_gts)[None, :, None], [batch_size, 1, num_preds]),
        cost_argmin[:, None],
    )
    update_mask = tf.logical_and(min_mask, duplicate_mask[:, None])
    matching_matrix = tf.where(
        update_mask, tf.ones_like(matching_matrix), matching_matrix)

    # Find the final matching and collect the matched targets.
    matched_gt_indices = tf.argmax(
        matching_matrix, axis=1, output_type=tf.int32
    )
    matched_mask = tf.reduce_sum(matching_matrix, axis=1) > 0
    matched_targets = tf.gather_nd(
        labels, matched_gt_indices[..., None], batch_dims=1
    )
    return indices, anchors, matched_mask, matched_targets, num_duplicates

  def _find_three_positives(self, labels, predictions):
    """Finds three matching anchors for each ground-truth."""
    label_shape = tf.shape(labels)
    batch_size, max_boxes = label_shape[0], label_shape[1]
    masks, indices, anch = [], [], []
    anchor_indices = tf.tile(
        tf.range(self._num_anchors, dtype=tf.float32)[None, None],
        [batch_size, max_boxes, 1],
    )
    # Append anchor indices to labels.
    labels = tf.tile(labels[:, :, None], [1, 1, self._num_anchors, 1])
    labels = tf.concat([labels, anchor_indices[..., None]], axis=-1)

    # Bias is used to determine the matching. 0.5 means matching anchors that
    # fall in the 0.5 differences in the feature map. For instance, a box
    # coordinates of (15.6, 35.4) will match the anchors at [15, 35], [16, 35],
    # and [15, 34].
    bias = 0.5  # bias
    off = (
        tf.constant(
            [
                [0, 0],
                [1, 0], [0, 1], [-1, 0], [0, -1],  # j, k, l, m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            tf.float32,
        )
        * bias
    )  # offsets

    for i in range(self._num_layers):
      anchors = self._anchors[i]
      _, _, h, w, _ = predictions[str(i + 3)].get_shape().as_list()
      gain = tf.constant([1, w, h, w, h, 1], dtype=tf.float32)

      t = labels * gain

      # Filter out targets that do not match the current anchors.
      wh_ratio = t[..., 3:5] / tf.cast(anchors[None, None], tf.float32)
      labels_mask = tf.less(
          tf.reduce_max(tf.maximum(wh_ratio, 1.0 / wh_ratio), axis=-1),
          self._anchor_threshold,
      )[..., None]
      # Compute valid mask for ground-truths.
      labels_mask = tf.logical_and(t[..., :1] != -1, labels_mask)

      labels_mask = tf.reshape(labels_mask, [batch_size, -1])
      t = tf.reshape(t, [batch_size, -1, 6])

      # Find the matching offsets for valid labels.
      gxy = t[..., 1:3]  # grid xy
      gxi = gain[1:3] - gxy  # inverse
      j, k = tf.split((gxy % 1.0 < bias) & (gxy >= 1.0), 2, axis=-1)
      l, m = tf.split((gxi % 1.0 < bias) & (gxi >= 1.0), 2, axis=-1)

      j, k, l, m = j[..., 0], k[..., 0], l[..., 0], m[..., 0]

      # Note that j and l, k and m are conjugate to each other, so at most one
      # of them will be True during running. Therefore, we can reduce memory
      # usage by gathering the selected index.
      x_map = tf.cast(tf.stack([j, l], axis=-1), tf.int8)
      y_map = tf.cast(tf.stack([k, m], axis=-1), tf.int8)

      # Add the indices offsets.
      x_indices = tf.argmax(x_map, axis=-1) * 2 + 1
      y_indices = tf.argmax(y_map, axis=-1) * 2 + 2
      three_targets_indices = tf.stack(
          [tf.zeros_like(x_indices), x_indices, y_indices], axis=-1
      )[..., None]

      # Gather the selected 3 targets from the 5-target map.
      j = tf.stack([tf.ones_like(j), j, k, l, m], axis=-1)
      three_targets_mask = tf.gather_nd(j, three_targets_indices, batch_dims=2)

      labels_mask = tf.tile(labels_mask[:, :, None], [1, 1, 5])
      t = tf.tile(t[:, :, None], [1, 1, 5, 1])

      labels_mask = tf.gather_nd(
          labels_mask, three_targets_indices, batch_dims=2
      )
      t = tf.gather_nd(t, three_targets_indices, batch_dims=2)

      offsets = tf.zeros_like(gxy)[:, :, None] + off[None, None]
      offsets = tf.gather_nd(offsets, three_targets_indices, batch_dims=2)

      gxy = t[..., 1:3]
      # Find the actual grid locations.
      gij = tf.cast(gxy - offsets * 2, tf.int32)
      gi, gj = tf.split(gij, 2, axis=-1)
      gi, gj = gi[..., 0], gj[..., 0]

      # Append the result.
      anchor_idx = tf.cast(t[..., 5], tf.int32)
      gain = tf.cast(gain, tf.int32)
      gi = tf.clip_by_value(gi, 0, gain[2] - 1)
      gj = tf.clip_by_value(gj, 0, gain[3] - 1)

      labels_mask = tf.logical_and(labels_mask, three_targets_mask)
      masks.append(labels_mask)
      indices.append(tf.stack([anchor_idx, gj, gi], axis=-1))
      anch.append(tf.gather(anchors, anchor_idx))

    # [batch_size, num_layers, num_anchors * max_boxes, num_targets]
    masks = tf.stack(masks, axis=1)
    indices = tf.stack(indices, axis=1)
    anch = tf.stack(anch, axis=1)
    return masks, indices, anch

  def report_stats(self):
    return {
        'num_gts': self._num_gts,
        'num_matchings': self._num_matchings,
        'num_duplicates': self._num_duplicates,
    }

  def report_separate_losses(self):
    """Returns separate losses that construct the reported loss."""
    return {
        'iou': self._iou,
        'box_loss': self._box_loss,
        'obj_loss': self._obj_loss,
        'cls_loss': self._cls_loss,
    }

  def get_config(self):
    """Configs for the loss constructor."""
    config = {
        'alpha': self._alpha,
        'gamma': self._gamma,
        'box_weight': self._box_weight,
        'obj_weight': self._obj_weight,
        'cls_weight': self._cls_weight,
        'iou_weight': self._iou_weight,
        'iou_mix_ratio': self._iou_mix_ratio,
        'pos_targets': self._pos_targets,
        'neg_targets': self._neg_targets,
        'num_classes': self._num_classes,
        'num_layers': self._num_layers,
        'num_anchors': self._num_anchors,
        'auto_balance': self._auto_balance,
        'balance': self._balance,
        'strides': self._strides,
        'anchors': self._anchors,
        'input_size': self._input_size,
        'anchor_threshold': self._anchor_threshold,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

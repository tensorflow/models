# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Contains common building blocks for yolo layer (detection layer)."""
from typing import Optional
import tensorflow as tf

from official.projects.yolo.losses import yolo_loss
from official.projects.yolo.ops import box_ops
from official.projects.yolo.ops import loss_utils
from official.vision.modeling.layers import detection_generator


class YoloLayer(tf.keras.layers.Layer):
  """Yolo layer (detection generator)."""

  def __init__(
      self,
      anchors,
      classes,
      iou_thresh=0.0,
      ignore_thresh=0.7,
      truth_thresh=1.0,
      nms_thresh=0.6,
      max_delta=10.0,
      loss_type='ciou',
      iou_normalizer=1.0,
      cls_normalizer=1.0,
      object_normalizer=1.0,
      use_scaled_loss=False,
      update_on_repeat=False,
      pre_nms_points=5000,
      label_smoothing=0.0,
      max_boxes=200,
      box_type='original',
      path_scale=None,
      scale_xy=None,
      nms_version='greedy',
      objectness_smooth=False,
      use_class_agnostic_nms: Optional[bool] = False,
      **kwargs
  ):
    """Parameters for the loss functions used at each detection head output.

    Args:
      anchors: `List[List[int]]` for the anchor boxes that are used in the
        model.
      classes: `int` for the number of classes.
      iou_thresh: `float` to use many anchors per object if IoU(Obj, Anchor) >
        iou_thresh.
      ignore_thresh: `float` for the IOU value over which the loss is not
        propagated, and a detection is assumed to have been made.
      truth_thresh: `float` for the IOU value over which the loss is propagated
        despite a detection being made'.
      nms_thresh: `float` for the minimum IOU value for an overlap.
      max_delta: gradient clipping to apply to the box loss.
      loss_type: `str` for the typeof iou loss to use with in {ciou, diou, giou,
        iou}.
      iou_normalizer: `float` for how much to scale the loss on the IOU or the
        boxes.
      cls_normalizer: `float` for how much to scale the loss on the classes.
      object_normalizer: `float` for how much to scale loss on the detection
        map.
      use_scaled_loss: `bool` for whether to use the scaled loss or the
        traditional loss.
      update_on_repeat: `bool` indicating how you would like to handle repeated
        indexes in a given [j, i] index. Setting this to True will give more
        consistent MAP, setting it to falls will improve recall by 1-2% but will
        sacrifice some MAP.
      pre_nms_points: `int` number of top candidate detections per class before
        NMS.
      label_smoothing: `float` for how much to smooth the loss on the classes.
      max_boxes: `int` for the maximum number of boxes retained over all
        classes.
      box_type: `str`, there are 3 different box types that will affect training
        differently {original, scaled and anchor_free}. The original method
        decodes the boxes by applying an exponential to the model width and
        height maps, then scaling the maps by the anchor boxes. This method is
        used in Yolo-v4, Yolo-v3, and all its counterparts. The Scale method
        squares the width and height and scales both by a fixed factor of 4.
        This method is used in the Scale Yolo models, as well as Yolov4-CSP.
        Finally, anchor_free is like the original method but will not apply an
        activation function to the boxes, this is used for some of the newer
        anchor free versions of YOLO.
      path_scale: `dict` for the size of the input tensors. Defaults to
        precalulated values from the `mask`.
      scale_xy: dictionary `float` values inidcating how far each pixel can see
        outside of its containment of 1.0. a value of 1.2 indicates there is a
        20% extended radius around each pixel that this specific pixel can
        predict values for a center at. the center can range from 0 - value/2 to
        1 + value/2, this value is set in the yolo filter, and resused here.
        there should be one value for scale_xy for each level from min_level to
        max_level.
      nms_version: `str` for which non max suppression to use.
      objectness_smooth: `float` for how much to smooth the loss on the
        detection map.
      use_class_agnostic_nms: A `bool` of whether non max suppression is
        operated on all the boxes using max scores across all classes. Only
        valid when nms_version is v2.
      **kwargs: Addtional keyword arguments.
    """
    super().__init__(**kwargs)
    self._anchors = anchors
    self._thresh = iou_thresh
    self._ignore_thresh = ignore_thresh
    self._truth_thresh = truth_thresh
    self._iou_normalizer = iou_normalizer
    self._cls_normalizer = cls_normalizer
    self._object_normalizer = object_normalizer
    self._objectness_smooth = objectness_smooth
    self._nms_thresh = nms_thresh
    self._max_boxes = max_boxes
    self._max_delta = max_delta
    self._classes = classes
    self._loss_type = loss_type
    self._use_class_agnostic_nms = use_class_agnostic_nms

    self._use_scaled_loss = use_scaled_loss
    self._update_on_repeat = update_on_repeat

    self._pre_nms_points = pre_nms_points
    self._label_smoothing = label_smoothing

    self._keys = list(anchors.keys())
    self._len_keys = len(self._keys)
    self._box_type = box_type
    self._path_scale = path_scale or {key: 2**int(key) for key in self._keys}

    self._nms_version = nms_version
    self._scale_xy = scale_xy or {key: 1.0 for key, _ in anchors.items()}

    self._generator = {}
    self._len_mask = {}
    for key in self._keys:
      anchors = self._anchors[key]
      self._generator[key] = loss_utils.GridGenerator(
          anchors, scale_anchors=self._path_scale[key])
      self._len_mask[key] = len(anchors)
    return

  def parse_prediction_path(self, key, inputs):
    shape_ = tf.shape(inputs)
    shape = inputs.get_shape().as_list()
    batchsize, height, width = shape_[0], shape[1], shape[2]

    if height is None or width is None:
      height, width = shape_[1], shape_[2]

    generator = self._generator[key]
    len_mask = self._len_mask[key]
    scale_xy = self._scale_xy[key]

    # Reshape the yolo output to (batchsize,
    #                             width,
    #                             height,
    #                             number_anchors,
    #                             remaining_points)
    data = tf.reshape(inputs, [-1, height, width, len_mask, self._classes + 5])

    # Use the grid generator to get the formatted anchor boxes and grid points
    # in shape [1, height, width, 2].
    centers, anchors = generator(height, width, batchsize, dtype=data.dtype)

    # Split the yolo detections into boxes, object score map, classes.
    boxes, obns_scores, class_scores = tf.split(
        data, [4, 1, self._classes], axis=-1)

    # Determine the number of classes.
    classes = class_scores.get_shape().as_list()[-1]

    # Configurable to use the new coordinates in scaled Yolo v4 or not.
    _, _, boxes = loss_utils.get_predicted_box(
        tf.cast(height, data.dtype),
        tf.cast(width, data.dtype),
        boxes,
        anchors,
        centers,
        scale_xy,
        stride=self._path_scale[key],
        darknet=False,
        box_type=self._box_type[key])

    # Convert boxes from yolo(x, y, w. h) to tensorflow(ymin, xmin, ymax, xmax).
    boxes = box_ops.xcycwh_to_yxyx(boxes)

    # Activate and detection map
    obns_scores = tf.math.sigmoid(obns_scores)

    # Convert detection map to class detection probabilities.
    class_scores = tf.math.sigmoid(class_scores) * obns_scores

    # Flatten predictions to [batchsize, N, -1] for non max supression.
    fill = height * width * len_mask
    boxes = tf.reshape(boxes, [-1, fill, 4])
    class_scores = tf.reshape(class_scores, [-1, fill, classes])
    obns_scores = tf.reshape(obns_scores, [-1, fill])
    return obns_scores, boxes, class_scores

  def __call__(self, inputs):
    boxes = []
    class_scores = []
    object_scores = []
    levels = list(inputs.keys())
    min_level = int(min(levels))
    max_level = int(max(levels))

    # Aggregate boxes over each scale.
    for i in range(min_level, max_level + 1):
      key = str(i)
      object_scores_, boxes_, class_scores_ = self.parse_prediction_path(
          key, inputs[key])
      boxes.append(boxes_)
      class_scores.append(class_scores_)
      object_scores.append(object_scores_)

    # Collate all predicitons.
    boxes = tf.concat(boxes, axis=1)
    object_scores = tf.concat(object_scores, axis=1)
    class_scores = tf.concat(class_scores, axis=1)

    # Get masks to threshold all the predicitons.
    object_mask = tf.cast(object_scores > self._thresh, object_scores.dtype)
    class_mask = tf.cast(class_scores > self._thresh, class_scores.dtype)

    # Apply thresholds mask to all the predictions.
    object_scores *= object_mask
    class_scores *= (tf.expand_dims(object_mask, axis=-1) * class_mask)

    # Make a copy of the original dtype.
    dtype = object_scores.dtype

    # Apply nms.
    if self._nms_version == 'greedy':
      # Greedy NMS.
      boxes, object_scores, class_scores, num_detections = (
          tf.image.combined_non_max_suppression(
              tf.expand_dims(tf.cast(boxes, dtype=tf.float32), axis=-2),
              tf.cast(class_scores, dtype=tf.float32),
              self._pre_nms_points,
              self._max_boxes,
              iou_threshold=self._nms_thresh,
              score_threshold=self._thresh,
          )
      )
    elif self._nms_version == 'v1':
      (boxes, object_scores, class_scores, num_detections, _) = (
          detection_generator._generate_detections_v1(  # pylint:disable=protected-access
              tf.expand_dims(tf.cast(boxes, dtype=tf.float32), axis=-2),
              tf.cast(class_scores, dtype=tf.float32),
              pre_nms_top_k=self._pre_nms_points,
              max_num_detections=self._max_boxes,
              nms_iou_threshold=self._nms_thresh,
              pre_nms_score_threshold=self._thresh,
          )
      )

    elif self._nms_version == 'v2' or self._nms_version == 'iou':
      (boxes, object_scores, class_scores, num_detections) = (
          detection_generator._generate_detections_v2(  # pylint:disable=protected-access
              tf.expand_dims(tf.cast(boxes, dtype=tf.float32), axis=-2),
              tf.cast(class_scores, dtype=tf.float32),
              pre_nms_top_k=self._pre_nms_points,
              max_num_detections=self._max_boxes,
              nms_iou_threshold=self._nms_thresh,
              pre_nms_score_threshold=self._thresh,
              use_class_agnostic_nms=self._use_class_agnostic_nms,
          )
      )

    # Cast the boxes and predicitons back to original datatype.
    boxes = tf.cast(boxes, dtype)
    class_scores = tf.cast(class_scores, dtype)
    object_scores = tf.cast(object_scores, dtype)

    # Format and return
    return {
        'bbox': boxes,
        'classes': class_scores,
        'confidence': object_scores,
        'num_detections': num_detections,
    }

  def get_losses(self):
    """Generates a dictionary of losses to apply to each path.

    Done in the detection generator because all parameters are the same
    across both loss and detection generator.

    Returns:
      Dict[str, tf.Tensor] of losses
    """
    loss = yolo_loss.YoloLoss(
        keys=self._keys,
        classes=self._classes,
        anchors=self._anchors,
        path_strides=self._path_scale,
        truth_thresholds=self._truth_thresh,
        ignore_thresholds=self._ignore_thresh,
        loss_types=self._loss_type,
        iou_normalizers=self._iou_normalizer,
        cls_normalizers=self._cls_normalizer,
        object_normalizers=self._object_normalizer,
        objectness_smooths=self._objectness_smooth,
        box_types=self._box_type,
        max_deltas=self._max_delta,
        scale_xys=self._scale_xy,
        use_scaled_loss=self._use_scaled_loss,
        update_on_repeat=self._update_on_repeat,
        label_smoothing=self._label_smoothing)
    return loss

  def get_config(self):
    return {
        'anchors': [list(a) for a in self._anchors],
        'thresh': self._thresh,
        'max_boxes': self._max_boxes,
    }

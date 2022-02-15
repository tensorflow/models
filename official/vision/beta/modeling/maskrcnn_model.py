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

"""R-CNN(-RS) models."""

from typing import Any, List, Mapping, Optional, Tuple, Union

import tensorflow as tf

from official.vision.beta.ops import anchor
from official.vision.beta.ops import box_ops


@tf.keras.utils.register_keras_serializable(package='Beta')
class MaskRCNNModel(tf.keras.Model):
  """The Mask R-CNN(-RS) and Cascade RCNN-RS models."""

  def __init__(self,
               backbone: tf.keras.Model,
               decoder: tf.keras.Model,
               rpn_head: tf.keras.layers.Layer,
               detection_head: Union[tf.keras.layers.Layer,
                                     List[tf.keras.layers.Layer]],
               roi_generator: tf.keras.layers.Layer,
               roi_sampler: Union[tf.keras.layers.Layer,
                                  List[tf.keras.layers.Layer]],
               roi_aligner: tf.keras.layers.Layer,
               detection_generator: tf.keras.layers.Layer,
               mask_head: Optional[tf.keras.layers.Layer] = None,
               mask_sampler: Optional[tf.keras.layers.Layer] = None,
               mask_roi_aligner: Optional[tf.keras.layers.Layer] = None,
               class_agnostic_bbox_pred: bool = False,
               cascade_class_ensemble: bool = False,
               min_level: Optional[int] = None,
               max_level: Optional[int] = None,
               num_scales: Optional[int] = None,
               aspect_ratios: Optional[List[float]] = None,
               anchor_size: Optional[float] = None,
               **kwargs):
    """Initializes the R-CNN(-RS) model.

    Args:
      backbone: `tf.keras.Model`, the backbone network.
      decoder: `tf.keras.Model`, the decoder network.
      rpn_head: the RPN head.
      detection_head: the detection head or a list of heads.
      roi_generator: the ROI generator.
      roi_sampler: a single ROI sampler or a list of ROI samplers for cascade
        detection heads.
      roi_aligner: the ROI aligner.
      detection_generator: the detection generator.
      mask_head: the mask head.
      mask_sampler: the mask sampler.
      mask_roi_aligner: the ROI alginer for mask prediction.
      class_agnostic_bbox_pred: if True, perform class agnostic bounding box
        prediction. Needs to be `True` for Cascade RCNN models.
      cascade_class_ensemble: if True, ensemble classification scores over all
        detection heads.
      min_level: Minimum level in output feature maps.
      max_level: Maximum level in output feature maps.
      num_scales: A number representing intermediate scales added on each level.
        For instances, num_scales=2 adds one additional intermediate anchor
        scales [2^0, 2^0.5] on each level.
      aspect_ratios: A list representing the aspect raito anchors added on each
        level. The number indicates the ratio of width to height. For instances,
        aspect_ratios=[1.0, 2.0, 0.5] adds three anchors on each scale level.
      anchor_size: A number representing the scale of size of the base anchor to
        the feature stride 2^level.
      **kwargs: keyword arguments to be passed.
    """
    super(MaskRCNNModel, self).__init__(**kwargs)
    self._config_dict = {
        'backbone': backbone,
        'decoder': decoder,
        'rpn_head': rpn_head,
        'detection_head': detection_head,
        'roi_generator': roi_generator,
        'roi_sampler': roi_sampler,
        'roi_aligner': roi_aligner,
        'detection_generator': detection_generator,
        'mask_head': mask_head,
        'mask_sampler': mask_sampler,
        'mask_roi_aligner': mask_roi_aligner,
        'class_agnostic_bbox_pred': class_agnostic_bbox_pred,
        'cascade_class_ensemble': cascade_class_ensemble,
        'min_level': min_level,
        'max_level': max_level,
        'num_scales': num_scales,
        'aspect_ratios': aspect_ratios,
        'anchor_size': anchor_size,
    }
    self.backbone = backbone
    self.decoder = decoder
    self.rpn_head = rpn_head
    if not isinstance(detection_head, (list, tuple)):
      self.detection_head = [detection_head]
    else:
      self.detection_head = detection_head
    self.roi_generator = roi_generator
    if not isinstance(roi_sampler, (list, tuple)):
      self.roi_sampler = [roi_sampler]
    else:
      self.roi_sampler = roi_sampler
    if len(self.roi_sampler) > 1 and not class_agnostic_bbox_pred:
      raise ValueError(
          '`class_agnostic_bbox_pred` needs to be True if multiple detection heads are specified.'
      )
    self.roi_aligner = roi_aligner
    self.detection_generator = detection_generator
    self._include_mask = mask_head is not None
    self.mask_head = mask_head
    if self._include_mask and mask_sampler is None:
      raise ValueError('`mask_sampler` is not provided in Mask R-CNN.')
    self.mask_sampler = mask_sampler
    if self._include_mask and mask_roi_aligner is None:
      raise ValueError('`mask_roi_aligner` is not provided in Mask R-CNN.')
    self.mask_roi_aligner = mask_roi_aligner
    # Weights for the regression losses for each FRCNN layer.
    # TODO(xianzhi): Make the weights configurable.
    self._cascade_layer_to_weights = [
        [10.0, 10.0, 5.0, 5.0],
        [20.0, 20.0, 10.0, 10.0],
        [30.0, 30.0, 15.0, 15.0],
    ]

  def call(self,
           images: tf.Tensor,
           image_shape: tf.Tensor,
           anchor_boxes: Optional[Mapping[str, tf.Tensor]] = None,
           gt_boxes: Optional[tf.Tensor] = None,
           gt_classes: Optional[tf.Tensor] = None,
           gt_masks: Optional[tf.Tensor] = None,
           training: Optional[bool] = None) -> Mapping[str, tf.Tensor]:

    model_outputs, intermediate_outputs = self._call_box_outputs(
        images=images, image_shape=image_shape, anchor_boxes=anchor_boxes,
        gt_boxes=gt_boxes, gt_classes=gt_classes, training=training)
    if not self._include_mask:
      return model_outputs

    model_mask_outputs = self._call_mask_outputs(
        model_box_outputs=model_outputs,
        features=model_outputs['decoder_features'],
        current_rois=intermediate_outputs['current_rois'],
        matched_gt_indices=intermediate_outputs['matched_gt_indices'],
        matched_gt_boxes=intermediate_outputs['matched_gt_boxes'],
        matched_gt_classes=intermediate_outputs['matched_gt_classes'],
        gt_masks=gt_masks,
        training=training)
    model_outputs.update(model_mask_outputs)
    return model_outputs

  def _get_backbone_and_decoder_features(self, images):

    backbone_features = self.backbone(images)
    if self.decoder:
      features = self.decoder(backbone_features)
    else:
      features = backbone_features
    return backbone_features, features

  def _call_box_outputs(
      self, images: tf.Tensor,
      image_shape: tf.Tensor,
      anchor_boxes: Optional[Mapping[str, tf.Tensor]] = None,
      gt_boxes: Optional[tf.Tensor] = None,
      gt_classes: Optional[tf.Tensor] = None,
      training: Optional[bool] = None) -> Tuple[
          Mapping[str, tf.Tensor], Mapping[str, tf.Tensor]]:
    """Implementation of the Faster-RCNN logic for boxes."""
    model_outputs = {}

    # Feature extraction.
    (backbone_features,
     decoder_features) = self._get_backbone_and_decoder_features(images)

    # Region proposal network.
    rpn_scores, rpn_boxes = self.rpn_head(decoder_features)

    model_outputs.update({
        'backbone_features': backbone_features,
        'decoder_features': decoder_features,
        'rpn_boxes': rpn_boxes,
        'rpn_scores': rpn_scores
    })

    # Generate anchor boxes for this batch if not provided.
    if anchor_boxes is None:
      _, image_height, image_width, _ = images.get_shape().as_list()
      anchor_boxes = anchor.Anchor(
          min_level=self._config_dict['min_level'],
          max_level=self._config_dict['max_level'],
          num_scales=self._config_dict['num_scales'],
          aspect_ratios=self._config_dict['aspect_ratios'],
          anchor_size=self._config_dict['anchor_size'],
          image_size=(image_height, image_width)).multilevel_boxes
      for l in anchor_boxes:
        anchor_boxes[l] = tf.tile(
            tf.expand_dims(anchor_boxes[l], axis=0),
            [tf.shape(images)[0], 1, 1, 1])

    # Generate RoIs.
    current_rois, _ = self.roi_generator(rpn_boxes, rpn_scores, anchor_boxes,
                                         image_shape, training)

    next_rois = current_rois
    all_class_outputs = []
    for cascade_num in range(len(self.roi_sampler)):
      # In cascade RCNN we want the higher layers to have different regression
      # weights as the predicted deltas become smaller and smaller.
      regression_weights = self._cascade_layer_to_weights[cascade_num]
      current_rois = next_rois

      (class_outputs, box_outputs, model_outputs, matched_gt_boxes,
       matched_gt_classes, matched_gt_indices,
       current_rois) = self._run_frcnn_head(
           features=decoder_features,
           rois=current_rois,
           gt_boxes=gt_boxes,
           gt_classes=gt_classes,
           training=training,
           model_outputs=model_outputs,
           cascade_num=cascade_num,
           regression_weights=regression_weights)
      all_class_outputs.append(class_outputs)

      # Generate ROIs for the next cascade head if there is any.
      if cascade_num < len(self.roi_sampler) - 1:
        next_rois = box_ops.decode_boxes(
            tf.cast(box_outputs, tf.float32),
            current_rois,
            weights=regression_weights)
        next_rois = box_ops.clip_boxes(next_rois,
                                       tf.expand_dims(image_shape, axis=1))

    if not training:
      if self._config_dict['cascade_class_ensemble']:
        class_outputs = tf.add_n(all_class_outputs) / len(all_class_outputs)

      detections = self.detection_generator(
          box_outputs,
          class_outputs,
          current_rois,
          image_shape,
          regression_weights,
          bbox_per_class=(not self._config_dict['class_agnostic_bbox_pred']))
      model_outputs.update({
          'cls_outputs': class_outputs,
          'box_outputs': box_outputs,
      })
      if self.detection_generator.get_config()['apply_nms']:
        model_outputs.update({
            'detection_boxes': detections['detection_boxes'],
            'detection_scores': detections['detection_scores'],
            'detection_classes': detections['detection_classes'],
            'num_detections': detections['num_detections']
        })
      else:
        model_outputs.update({
            'decoded_boxes': detections['decoded_boxes'],
            'decoded_box_scores': detections['decoded_box_scores']
        })

    intermediate_outputs = {
        'matched_gt_boxes': matched_gt_boxes,
        'matched_gt_indices': matched_gt_indices,
        'matched_gt_classes': matched_gt_classes,
        'current_rois': current_rois,
    }
    return (model_outputs, intermediate_outputs)

  def _call_mask_outputs(
      self,
      model_box_outputs: Mapping[str, tf.Tensor],
      features: tf.Tensor,
      current_rois: tf.Tensor,
      matched_gt_indices: tf.Tensor,
      matched_gt_boxes: tf.Tensor,
      matched_gt_classes: tf.Tensor,
      gt_masks: tf.Tensor,
      training: Optional[bool] = None) -> Mapping[str, tf.Tensor]:
    """Implementation of Mask-RCNN mask prediction logic."""

    model_outputs = dict(model_box_outputs)
    if training:
      current_rois, roi_classes, roi_masks = self.mask_sampler(
          current_rois, matched_gt_boxes, matched_gt_classes,
          matched_gt_indices, gt_masks)
      roi_masks = tf.stop_gradient(roi_masks)

      model_outputs.update({
          'mask_class_targets': roi_classes,
          'mask_targets': roi_masks,
      })
    else:
      current_rois = model_outputs['detection_boxes']
      roi_classes = model_outputs['detection_classes']

    mask_logits, mask_probs = self._features_to_mask_outputs(
        features, current_rois, roi_classes)

    if training:
      model_outputs.update({
          'mask_outputs': mask_logits,
      })
    else:
      model_outputs.update({
          'detection_masks': mask_probs,
      })
    return model_outputs

  def _run_frcnn_head(self, features, rois, gt_boxes, gt_classes, training,
                      model_outputs, cascade_num, regression_weights):
    """Runs the frcnn head that does both class and box prediction.

    Args:
      features: `list` of features from the feature extractor.
      rois: `list` of current rois that will be used to predict bbox refinement
        and classes from.
      gt_boxes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4].
        This tensor might have paddings with a negative value.
      gt_classes: [batch_size, MAX_INSTANCES] representing the groundtruth box
        classes. It is padded with -1s to indicate the invalid classes.
      training: `bool`, if model is training or being evaluated.
      model_outputs: `dict`, used for storing outputs used for eval and losses.
      cascade_num: `int`, the current frcnn layer in the cascade.
      regression_weights: `list`, weights used for l1 loss in bounding box
        regression.

    Returns:
      class_outputs: Class predictions for rois.
      box_outputs: Box predictions for rois. These are formatted for the
        regression loss and need to be converted before being used as rois
        in the next stage.
      model_outputs: Updated dict with predictions used for losses and eval.
      matched_gt_boxes: If `is_training` is true, then these give the gt box
        location of its positive match.
      matched_gt_classes: If `is_training` is true, then these give the gt class
         of the predicted box.
      matched_gt_boxes: If `is_training` is true, then these give the box
        location of its positive match.
      matched_gt_indices: If `is_training` is true, then gives the index of
        the positive box match. Used for mask prediction.
      rois: The sampled rois used for this layer.
    """
    # Only used during training.
    matched_gt_boxes, matched_gt_classes, matched_gt_indices = (None, None,
                                                                None)
    if training and gt_boxes is not None:
      rois = tf.stop_gradient(rois)

      current_roi_sampler = self.roi_sampler[cascade_num]
      rois, matched_gt_boxes, matched_gt_classes, matched_gt_indices = (
          current_roi_sampler(rois, gt_boxes, gt_classes))
      # Create bounding box training targets.
      box_targets = box_ops.encode_boxes(
          matched_gt_boxes, rois, weights=regression_weights)
      # If the target is background, the box target is set to all 0s.
      box_targets = tf.where(
          tf.tile(
              tf.expand_dims(tf.equal(matched_gt_classes, 0), axis=-1),
              [1, 1, 4]), tf.zeros_like(box_targets), box_targets)
      model_outputs.update({
          'class_targets_{}'.format(cascade_num)
          if cascade_num else 'class_targets':
              matched_gt_classes,
          'box_targets_{}'.format(cascade_num)
          if cascade_num else 'box_targets':
              box_targets,
      })

    # Get roi features.
    roi_features = self.roi_aligner(features, rois)

    # Run frcnn head to get class and bbox predictions.
    current_detection_head = self.detection_head[cascade_num]
    class_outputs, box_outputs = current_detection_head(roi_features)

    model_outputs.update({
        'class_outputs_{}'.format(cascade_num)
        if cascade_num else 'class_outputs':
            class_outputs,
        'box_outputs_{}'.format(cascade_num) if cascade_num else 'box_outputs':
            box_outputs,
    })
    return (class_outputs, box_outputs, model_outputs, matched_gt_boxes,
            matched_gt_classes, matched_gt_indices, rois)

  def _features_to_mask_outputs(self, features, rois, roi_classes):
    # Mask RoI align.
    mask_roi_features = self.mask_roi_aligner(features, rois)

    # Mask head.
    raw_masks = self.mask_head([mask_roi_features, roi_classes])

    return raw_masks, tf.nn.sigmoid(raw_masks)

  @property
  def checkpoint_items(
      self) -> Mapping[str, Union[tf.keras.Model, tf.keras.layers.Layer]]:
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(
        backbone=self.backbone,
        rpn_head=self.rpn_head,
        detection_head=self.detection_head)
    if self.decoder is not None:
      items.update(decoder=self.decoder)
    if self._include_mask:
      items.update(mask_head=self.mask_head)

    return items

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)

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

"""Mask R-CNN model."""

from typing import List, Mapping, Optional, Union

# Import libraries

from absl import logging
import tensorflow as tf

from official.vision.beta.modeling import maskrcnn_model


def resize_as(source, size):

  source = tf.transpose(source, (0, 2, 3, 1))
  source = tf.image.resize(source, (size, size))
  return tf.transpose(source, (0, 3, 1, 2))


class DeepMaskRCNNModel(maskrcnn_model.MaskRCNNModel):
  """The Mask R-CNN model."""

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
               use_gt_boxes_for_masks=False,
               **kwargs):
    """Initializes the Mask R-CNN model.

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
      use_gt_boxes_for_masks: bool, if set, crop using groundtruth boxes instead
        of proposals for training mask head
      **kwargs: keyword arguments to be passed.
    """
    super(DeepMaskRCNNModel, self).__init__(
        backbone=backbone,
        decoder=decoder,
        rpn_head=rpn_head,
        detection_head=detection_head,
        roi_generator=roi_generator,
        roi_sampler=roi_sampler,
        roi_aligner=roi_aligner,
        detection_generator=detection_generator,
        mask_head=mask_head,
        mask_sampler=mask_sampler,
        mask_roi_aligner=mask_roi_aligner,
        class_agnostic_bbox_pred=class_agnostic_bbox_pred,
        cascade_class_ensemble=cascade_class_ensemble,
        min_level=min_level,
        max_level=max_level,
        num_scales=num_scales,
        aspect_ratios=aspect_ratios,
        anchor_size=anchor_size,
        **kwargs)

    self._config_dict['use_gt_boxes_for_masks'] = use_gt_boxes_for_masks

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
        gt_classes=gt_classes,
        gt_boxes=gt_boxes,
        training=training)
    model_outputs.update(model_mask_outputs)
    return model_outputs

  def call_images_and_boxes(self, images, boxes):
    """Predict masks given an image and bounding boxes."""

    _, decoder_features = self._get_backbone_and_decoder_features(images)
    boxes_shape = tf.shape(boxes)
    batch_size, num_boxes = boxes_shape[0], boxes_shape[1]
    classes = tf.zeros((batch_size, num_boxes), dtype=tf.int32)

    _, mask_probs = self._features_to_mask_outputs(
        decoder_features, boxes, classes)
    return {
        'detection_masks': mask_probs
    }

  def _call_mask_outputs(
      self,
      model_box_outputs: Mapping[str, tf.Tensor],
      features: tf.Tensor,
      current_rois: tf.Tensor,
      matched_gt_indices: tf.Tensor,
      matched_gt_boxes: tf.Tensor,
      matched_gt_classes: tf.Tensor,
      gt_masks: tf.Tensor,
      gt_classes: tf.Tensor,
      gt_boxes: tf.Tensor,
      training: Optional[bool] = None) -> Mapping[str, tf.Tensor]:

    model_outputs = dict(model_box_outputs)
    if training:
      if self._config_dict['use_gt_boxes_for_masks']:
        mask_size = (
            self.mask_roi_aligner._config_dict['crop_size'] *  # pylint:disable=protected-access
            self.mask_head._config_dict['upsample_factor']  # pylint:disable=protected-access
        )
        gt_masks = resize_as(source=gt_masks, size=mask_size)

        logging.info('Using GT class and mask targets.')
        model_outputs.update({
            'mask_class_targets': gt_classes,
            'mask_targets': gt_masks,
        })
      else:
        rois, roi_classes, roi_masks = self.mask_sampler(
            current_rois, matched_gt_boxes, matched_gt_classes,
            matched_gt_indices, gt_masks)
        roi_masks = tf.stop_gradient(roi_masks)
        model_outputs.update({
            'mask_class_targets': roi_classes,
            'mask_targets': roi_masks,
        })

    else:
      rois = model_outputs['detection_boxes']
      roi_classes = model_outputs['detection_classes']

    # Mask RoI align.
    if training and self._config_dict['use_gt_boxes_for_masks']:
      logging.info('Using GT mask roi features.')
      roi_aligner_boxes = gt_boxes
      mask_head_classes = gt_classes

    else:
      roi_aligner_boxes = rois
      mask_head_classes = roi_classes

    mask_logits, mask_probs = self._features_to_mask_outputs(
        features, roi_aligner_boxes, mask_head_classes)

    if training:
      model_outputs.update({
          'mask_outputs': mask_logits,
      })
    else:
      model_outputs.update({
          'detection_masks': mask_probs,
      })
    return model_outputs

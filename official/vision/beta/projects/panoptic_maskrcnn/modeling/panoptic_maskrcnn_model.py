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

"""Panoptic Segmentation model."""

from typing import List, Mapping, Optional, Union

import tensorflow as tf

from official.vision.beta.modeling import maskrcnn_model


@tf.keras.utils.register_keras_serializable(package='Vision')
class PanopticMaskRCNNModel(maskrcnn_model.MaskRCNNModel):
  """The Panoptic Segmentation model."""

  def __init__(
      self,
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
      panoptic_segmentation_generator: Optional[tf.keras.layers.Layer] = None,
      mask_head: Optional[tf.keras.layers.Layer] = None,
      mask_sampler: Optional[tf.keras.layers.Layer] = None,
      mask_roi_aligner: Optional[tf.keras.layers.Layer] = None,
      segmentation_backbone: Optional[tf.keras.Model] = None,
      segmentation_decoder: Optional[tf.keras.Model] = None,
      segmentation_head: tf.keras.layers.Layer = None,
      class_agnostic_bbox_pred: bool = False,
      cascade_class_ensemble: bool = False,
      min_level: Optional[int] = None,
      max_level: Optional[int] = None,
      num_scales: Optional[int] = None,
      aspect_ratios: Optional[List[float]] = None,
      anchor_size: Optional[float] = None,  # pytype: disable=annotation-type-mismatch  # typed-keras
      **kwargs):
    """Initializes the Panoptic Mask R-CNN model.

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
      panoptic_segmentation_generator: the panoptic segmentation generator that
        is used to merge instance and semantic segmentation masks.
      mask_head: the mask head.
      mask_sampler: the mask sampler.
      mask_roi_aligner: the ROI alginer for mask prediction.
      segmentation_backbone: `tf.keras.Model`, the backbone network for the
        segmentation head for panoptic task. Providing `segmentation_backbone`
        will allow the segmentation head to use a standlone backbone. Setting
        `segmentation_backbone=None` would enable backbone sharing between the
        MaskRCNN model and segmentation head.
      segmentation_decoder: `tf.keras.Model`, the decoder network for the
        segmentation head for panoptic task. Providing `segmentation_decoder`
        will allow the segmentation head to use a standlone decoder. Setting
        `segmentation_decoder=None` would enable decoder sharing between the
        MaskRCNN model and segmentation head. Decoders can only be shared when
        `segmentation_backbone` is shared as well.
      segmentation_head: segmentatation head for panoptic task.
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
    super(PanopticMaskRCNNModel, self).__init__(
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

    self._config_dict.update({
        'segmentation_backbone': segmentation_backbone,
        'segmentation_decoder': segmentation_decoder,
        'segmentation_head': segmentation_head
    })

    if panoptic_segmentation_generator is not None:
      self._config_dict.update(
          {'panoptic_segmentation_generator': panoptic_segmentation_generator})

    if not self._include_mask:
      raise ValueError(
          '`mask_head` needs to be provided for Panoptic Mask R-CNN.')
    if segmentation_backbone is not None and segmentation_decoder is None:
      raise ValueError(
          '`segmentation_decoder` needs to be provided for Panoptic Mask R-CNN'
          'if `backbone` is not shared.')

    self.segmentation_backbone = segmentation_backbone
    self.segmentation_decoder = segmentation_decoder
    self.segmentation_head = segmentation_head
    self.panoptic_segmentation_generator = panoptic_segmentation_generator

  def call(self,
           images: tf.Tensor,
           image_info: tf.Tensor,
           anchor_boxes: Optional[Mapping[str, tf.Tensor]] = None,
           gt_boxes: Optional[tf.Tensor] = None,
           gt_classes: Optional[tf.Tensor] = None,
           gt_masks: Optional[tf.Tensor] = None,
           training: Optional[bool] = None) -> Mapping[str, tf.Tensor]:
    image_shape = image_info[:, 1, :]
    model_outputs = super(PanopticMaskRCNNModel, self).call(
        images=images,
        image_shape=image_shape,
        anchor_boxes=anchor_boxes,
        gt_boxes=gt_boxes,
        gt_classes=gt_classes,
        gt_masks=gt_masks,
        training=training)

    if self.segmentation_backbone is not None:
      backbone_features = self.segmentation_backbone(images, training=training)
    else:
      backbone_features = model_outputs['backbone_features']

    if self.segmentation_decoder is not None:
      decoder_features = self.segmentation_decoder(
          backbone_features, training=training)
    else:
      decoder_features = model_outputs['decoder_features']

    segmentation_outputs = self.segmentation_head(
        backbone_features, decoder_features, training=training)

    model_outputs.update({
        'segmentation_outputs': segmentation_outputs,
    })

    if not training and self.panoptic_segmentation_generator is not None:
      panoptic_outputs = self.panoptic_segmentation_generator(
          model_outputs, image_info=image_info)
      model_outputs.update({'panoptic_outputs': panoptic_outputs})

    return model_outputs

  @property
  def checkpoint_items(
      self) -> Mapping[str, Union[tf.keras.Model, tf.keras.layers.Layer]]:
    """Returns a dictionary of items to be additionally checkpointed."""
    items = super(PanopticMaskRCNNModel, self).checkpoint_items
    if self.segmentation_backbone is not None:
      items.update(segmentation_backbone=self.segmentation_backbone)
    if self.segmentation_decoder is not None:
      items.update(segmentation_decoder=self.segmentation_decoder)
    items.update(segmentation_head=self.segmentation_head)
    return items

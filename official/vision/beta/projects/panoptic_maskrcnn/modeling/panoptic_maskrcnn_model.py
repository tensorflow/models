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

# Import libraries
import tensorflow as tf

from official.vision.beta.modeling import maskrcnn_model
from official.vision.beta.ops import anchor
from official.vision.beta.ops import box_ops


@tf.keras.utils.register_keras_serializable(package='Vision')
class PanopticMaskRCNNModel(maskrcnn_model.MaskRCNNModel):
  """The Panoptic Segmentation model."""

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
               segmentation_backbone: Optional[tf.keras.Model] = None,
               segmentation_decoder: Optional[tf.keras.Model] = None,
               segmentation_head: tf.keras.layers.Layer = None,
               class_agnostic_bbox_pred: bool = False,
               cascade_class_ensemble: bool = False,
               min_level: Optional[int] = None,
               max_level: Optional[int] = None,
               num_scales: Optional[int] = None,
               aspect_ratios: Optional[List[float]] = None,
               anchor_size: Optional[float] = None,
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
      mask_head: the mask head.
      mask_sampler: the mask sampler.
      mask_roi_aligner: the ROI alginer for mask prediction.
      segmentation_backbone: `tf.keras.Model`, the backbone network for the
        segmentation head for panoptic task. Providing `segmentation_backbone`
        will allow the segmentation head to use a standlone backbone. Setting
        `segmentation_backbone=None` would enable backbone sharing between
        the MaskRCNN model and segmentation head.
      segmentation_decoder: `tf.keras.Model`, the decoder network for the
        segmentation head for panoptic task. Providing `segmentation_decoder`
        will allow the segmentation head to use a standlone decoder. Setting
        `segmentation_decoder=None` would enable decoder sharing between
        the MaskRCNN model and segmentation head. Decoders can only be shared
        when `segmentation_backbone` is shared as well. 
      segmentation_head: segmentatation head for panoptic task.
      class_agnostic_bbox_pred: if True, perform class agnostic bounding box
        prediction. Needs to be `True` for Cascade RCNN models.
      cascade_class_ensemble: if True, ensemble classification scores over
        all detection heads.
      min_level: Minimum level in output feature maps.
      max_level: Maximum level in output feature maps.
      num_scales: A number representing intermediate scales added
        on each level. For instances, num_scales=2 adds one additional
        intermediate anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: A list representing the aspect raito
        anchors added on each level. The number indicates the ratio of width to
        height. For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors
        on each scale level.
      anchor_size: A number representing the scale of size of the base
        anchor to the feature stride 2^level.
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
        'segmentation_backbone':segmentation_backbone,
        'segmentation_decoder': segmentation_decoder,
        'segmentation_head': segmentation_head
        })

    if not self._include_mask:
      raise ValueError(
          '`mask_head` needs to be provided for Panoptic Mask R-CNN.')
    if segmentation_backbone is not None and segmentation_decoder is None:
      raise ValueError(
          '`segmentation_decoder` needs to be provided for Panoptic Mask R-CNN if `backbone` is not shared.'
      )

    self.segmentation_backbone = segmentation_backbone
    self.segmentation_decoder = segmentation_decoder
    self.segmentation_head = segmentation_head

  def call(self,
           images: tf.Tensor,
           image_shape: tf.Tensor,
           anchor_boxes: Optional[Mapping[str, tf.Tensor]] = None,
           gt_boxes: Optional[tf.Tensor] = None,
           gt_classes: Optional[tf.Tensor] = None,
           gt_masks: Optional[tf.Tensor] = None,
           training: Optional[bool] = None) -> Mapping[str, tf.Tensor]:
    model_outputs = {}

    # Feature extraction.
    backbone_features = self.backbone(images)
    if self.decoder:
      decoder_features = self.decoder(backbone_features)

    # Region proposal network.
    rpn_scores, rpn_boxes = self.rpn_head(decoder_features)

    model_outputs.update({
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

    if not self._include_mask:
      return model_outputs

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

    # Mask RoI align.
    mask_roi_features = self.mask_roi_aligner(decoder_features, current_rois)

    # Mask head.
    raw_masks = self.mask_head([mask_roi_features, roi_classes])

    if training:
      model_outputs.update({
          'mask_outputs': raw_masks,
      })
    else:
      model_outputs.update({
          'detection_masks': tf.math.sigmoid(raw_masks),
      })

    if self.segmentation_backbone is not None:
      backbone_features = self.segmentation_backbone(
        images,
        training=training)

    if self.segmentation_decoder is not None:
      decoder_features = self.segmentation_decoder(
        backbone_features,
        training=training)
    
    segmentation_outputs = self.segmentation_head(
      backbone_features,
      decoder_features,
      training=training)

    model_outputs.update({
        'segmentation_outputs': segmentation_outputs,
    })

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

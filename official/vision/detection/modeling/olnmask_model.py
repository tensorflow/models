# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Model defination for the Object Localization Network (OLN) Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from official.vision.detection.dataloader import anchor
from official.vision.detection.dataloader import mode_keys
from official.vision.detection.modeling import losses
from official.vision.detection.modeling.architecture import factory
from official.vision.detection.modeling.architecture import keras_utils
from official.vision.detection.modeling.maskrcnn_model import MaskrcnnModel
from official.vision.detection.ops import postprocess_ops
from official.vision.detection.ops import roi_ops
from official.vision.detection.ops import spatial_transform_ops
from official.vision.detection.ops import target_ops
from official.vision.detection.utils import box_utils


class OlnMaskModel(MaskrcnnModel):
  """OLN-Mask model function."""

  def __init__(self, params):
    super(OlnMaskModel, self).__init__(params)

    self._params = params

    # Different heads and layers.
    self._include_rpn_class = params.architecture.include_rpn_class
    self._include_mask = params.architecture.include_mask
    self._include_frcnn_class = params.architecture.include_frcnn_class
    self._include_frcnn_box = params.architecture.include_frcnn_box
    self._include_centerness = params.rpn_head.has_centerness
    self._include_box_score = (params.frcnn_head.has_scoring and
                               params.architecture.include_frcnn_box)
    self._include_mask_score = (params.mrcnn_head.has_scoring and
                                params.architecture.include_mask)

    # Architecture generators.
    self._backbone_fn = factory.backbone_generator(params)
    self._fpn_fn = factory.multilevel_features_generator(params)
    self._rpn_head_fn = factory.rpn_head_generator(params)
    if self._include_centerness:
      self._rpn_head_fn = factory.oln_rpn_head_generator(params)
    else:
      self._rpn_head_fn = factory.rpn_head_generator(params)
    self._generate_rois_fn = roi_ops.OlnROIGenerator(params.roi_proposal)
    self._sample_rois_fn = target_ops.ROIScoreSampler(params.roi_sampling)
    self._sample_masks_fn = target_ops.MaskSampler(
        params.architecture.mask_target_size,
        params.mask_sampling.num_mask_samples_per_image)

    if self._include_box_score:
      self._frcnn_head_fn = factory.oln_box_score_head_generator(params)
    else:
      self._frcnn_head_fn = factory.fast_rcnn_head_generator(params)

    if self._include_mask:
      if self._include_mask_score:
        self._mrcnn_head_fn = factory.oln_mask_score_head_generator(params)
      else:
        self._mrcnn_head_fn = factory.mask_rcnn_head_generator(params)

    # Loss function.
    self._rpn_score_loss_fn = losses.RpnScoreLoss(params.rpn_score_loss)
    self._rpn_box_loss_fn = losses.RpnBoxLoss(params.rpn_box_loss)
    if self._include_centerness:
      self._rpn_iou_loss_fn = losses.OlnRpnIoULoss()
      self._rpn_center_loss_fn = losses.OlnRpnCenterLoss()
    self._frcnn_class_loss_fn = losses.FastrcnnClassLoss()
    self._frcnn_box_loss_fn = losses.FastrcnnBoxLoss(params.frcnn_box_loss)
    if self._include_box_score:
      self._frcnn_box_score_loss_fn = losses.OlnBoxScoreLoss(
          params.frcnn_box_score_loss)
    if self._include_mask:
      self._mask_loss_fn = losses.MaskrcnnLoss()

    self._generate_detections_fn = postprocess_ops.OlnDetectionGenerator(
        params.postprocess)

    self._transpose_input = params.train.transpose_input
    assert not self._transpose_input, 'Transpose input is not supportted.'

  def build_outputs(self, inputs, mode):
    is_training = mode == mode_keys.TRAIN
    model_outputs = {}

    image = inputs['image']
    _, image_height, image_width, _ = image.get_shape().as_list()
    backbone_features = self._backbone_fn(image, is_training)
    fpn_features = self._fpn_fn(backbone_features, is_training)

    # rpn_centerness.
    if self._include_centerness:
      rpn_score_outputs, rpn_box_outputs, rpn_center_outputs = (
          self._rpn_head_fn(fpn_features, is_training))
      model_outputs.update({
          'rpn_center_outputs':
              tf.nest.map_structure(lambda x: tf.cast(x, tf.float32),
                                    rpn_center_outputs),
      })
      object_scores = rpn_center_outputs
    else:
      rpn_score_outputs, rpn_box_outputs = self._rpn_head_fn(
          fpn_features, is_training)
      object_scores = None
    model_outputs.update({
        'rpn_score_outputs':
            tf.nest.map_structure(lambda x: tf.cast(x, tf.float32),
                                  rpn_score_outputs),
        'rpn_box_outputs':
            tf.nest.map_structure(lambda x: tf.cast(x, tf.float32),
                                  rpn_box_outputs),
    })
    input_anchor = anchor.Anchor(self._params.architecture.min_level,
                                 self._params.architecture.max_level,
                                 self._params.anchor.num_scales,
                                 self._params.anchor.aspect_ratios,
                                 self._params.anchor.anchor_size,
                                 (image_height, image_width))
    rpn_rois, rpn_roi_scores = self._generate_rois_fn(
        rpn_box_outputs,
        rpn_score_outputs,
        input_anchor.multilevel_boxes,
        inputs['image_info'][:, 1, :],
        is_training,
        is_box_lrtb=self._include_centerness,
        object_scores=object_scores,
        )
    if (not self._include_frcnn_class and
        not self._include_frcnn_box and
        not self._include_mask):
      # if not is_training:
      # For direct RPN detection,
      # use dummy box_outputs = (dy,dx,dh,dw = 0,0,0,0)
      box_outputs = tf.zeros_like(rpn_rois)
      box_outputs = tf.concat([box_outputs, box_outputs], -1)
      boxes, scores, classes, valid_detections = self._generate_detections_fn(
          box_outputs, rpn_roi_scores, rpn_rois,
          inputs['image_info'][:, 1:2, :],
          is_single_fg_score=True,  # if no_background, no softmax is applied.
          keep_nms=True)
      model_outputs.update({
          'num_detections': valid_detections,
          'detection_boxes': boxes,
          'detection_classes': classes,
          'detection_scores': scores,
      })
      return model_outputs

    # ---- OLN-Proposal finishes here. ----

    if is_training:
      rpn_rois = tf.stop_gradient(rpn_rois)
      rpn_roi_scores = tf.stop_gradient(rpn_roi_scores)

      # Sample proposals.
      (rpn_rois, rpn_roi_scores, matched_gt_boxes, matched_gt_classes,
       matched_gt_indices) = (
           self._sample_rois_fn(rpn_rois, rpn_roi_scores, inputs['gt_boxes'],
                                inputs['gt_classes']))
      # Create bounding box training targets.
      box_targets = box_utils.encode_boxes(
          matched_gt_boxes, rpn_rois, weights=[10.0, 10.0, 5.0, 5.0])
      # If the target is background, the box target is set to all 0s.
      box_targets = tf.where(
          tf.tile(
              tf.expand_dims(tf.equal(matched_gt_classes, 0), axis=-1),
              [1, 1, 4]), tf.zeros_like(box_targets), box_targets)
      model_outputs.update({
          'class_targets': matched_gt_classes,
          'box_targets': box_targets,
      })
      # Create Box-IoU targets. {
      box_ious = box_utils.bbox_overlap(
          rpn_rois, inputs['gt_boxes'])
      matched_box_ious = tf.reduce_max(box_ious, 2)
      model_outputs.update({
          'box_iou_targets': matched_box_ious,})  # }

    roi_features = spatial_transform_ops.multilevel_crop_and_resize(
        fpn_features, rpn_rois, output_size=7)

    if not self._include_box_score:
      class_outputs, box_outputs = self._frcnn_head_fn(
          roi_features, is_training)
    else:
      class_outputs, box_outputs, score_outputs = self._frcnn_head_fn(
          roi_features, is_training)
      model_outputs.update({
          'box_score_outputs':
              tf.nest.map_structure(lambda x: tf.cast(x, tf.float32),
                                    score_outputs),})
    model_outputs.update({
        'class_outputs':
            tf.nest.map_structure(lambda x: tf.cast(x, tf.float32),
                                  class_outputs),
        'box_outputs':
            tf.nest.map_structure(lambda x: tf.cast(x, tf.float32),
                                  box_outputs),
    })

    # Add this output to train to make the checkpoint loadable in predict mode.
    # If we skip it in train mode, the heads will be out-of-order and checkpoint
    # loading will fail.
    if not self._include_frcnn_box:
      box_outputs = tf.zeros_like(box_outputs)  # dummy zeros.

    if self._include_box_score:
      score_outputs = tf.cast(tf.squeeze(score_outputs, -1),
                              rpn_roi_scores.dtype)

      # box-score = (rpn-centerness * box-iou)^(1/2)
      # TR: rpn_roi_scores: b,1000, score_outputs: b,512
      # TS: rpn_roi_scores: b,1000, score_outputs: b,1000
      box_scores = tf.pow(
          rpn_roi_scores * tf.sigmoid(score_outputs), 1/2.)

    if not self._include_frcnn_class:
      boxes, scores, classes, valid_detections = self._generate_detections_fn(
          box_outputs,
          box_scores,
          rpn_rois,
          inputs['image_info'][:, 1:2, :],
          is_single_fg_score=True,
          keep_nms=True,)
    else:
      boxes, scores, classes, valid_detections = self._generate_detections_fn(
          box_outputs, class_outputs, rpn_rois,
          inputs['image_info'][:, 1:2, :],
          keep_nms=True,)
    model_outputs.update({
        'num_detections': valid_detections,
        'detection_boxes': boxes,
        'detection_classes': classes,
        'detection_scores': scores,
    })

    # ---- OLN-Box finishes here. ----

    if not self._include_mask:
      return model_outputs

    if is_training:
      rpn_rois, classes, mask_targets = self._sample_masks_fn(
          rpn_rois, matched_gt_boxes, matched_gt_classes, matched_gt_indices,
          inputs['gt_masks'])
      mask_targets = tf.stop_gradient(mask_targets)

      classes = tf.cast(classes, dtype=tf.int32)

      model_outputs.update({
          'mask_targets': mask_targets,
          'sampled_class_targets': classes,
      })
    else:
      rpn_rois = boxes
      classes = tf.cast(classes, dtype=tf.int32)

    mask_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
        fpn_features, rpn_rois, output_size=14)

    mask_outputs = self._mrcnn_head_fn(mask_roi_features, classes, is_training)

    if is_training:
      model_outputs.update({
          'mask_outputs':
              tf.nest.map_structure(lambda x: tf.cast(x, tf.float32),
                                    mask_outputs),
      })
    else:
      model_outputs.update({'detection_masks': tf.nn.sigmoid(mask_outputs)})

    return model_outputs

  def build_loss_fn(self):
    if self._keras_model is None:
      raise ValueError('build_loss_fn() must be called after build_model().')

    filter_fn = self.make_filter_trainable_variables_fn()
    trainable_variables = filter_fn(self._keras_model.trainable_variables)

    def _total_loss_fn(labels, outputs):
      if self._include_rpn_class:
        rpn_score_loss = self._rpn_score_loss_fn(outputs['rpn_score_outputs'],
                                                 labels['rpn_score_targets'])
      else:
        rpn_score_loss = 0.0
      if self._include_centerness:
        rpn_center_loss = self._rpn_center_loss_fn(
            outputs['rpn_center_outputs'], labels['rpn_center_targets'])
        rpn_box_loss = self._rpn_iou_loss_fn(
            outputs['rpn_box_outputs'], labels['rpn_box_targets'],
            labels['rpn_center_targets'])
      else:
        rpn_center_loss = 0.0
        rpn_box_loss = self._rpn_box_loss_fn(
            outputs['rpn_box_outputs'], labels['rpn_box_targets'])

      if self._include_frcnn_class:
        frcnn_class_loss = self._frcnn_class_loss_fn(
            outputs['class_outputs'], outputs['class_targets'])
      else:
        frcnn_class_loss = 0.0
      if self._include_frcnn_box:
        frcnn_box_loss = self._frcnn_box_loss_fn(
            outputs['box_outputs'], outputs['class_targets'],
            outputs['box_targets'])
      else:
        frcnn_box_loss = 0.0
      if self._include_box_score:
        box_score_loss = self._frcnn_box_score_loss_fn(
            outputs['box_score_outputs'], outputs['box_iou_targets'])
      else:
        box_score_loss = 0.0

      if self._include_mask:
        mask_loss = self._mask_loss_fn(outputs['mask_outputs'],
                                       outputs['mask_targets'],
                                       outputs['sampled_class_targets'])
      else:
        mask_loss = 0.0

      model_loss = (
          rpn_score_loss + rpn_box_loss + rpn_center_loss +
          frcnn_class_loss + frcnn_box_loss + box_score_loss +
          mask_loss)

      l2_regularization_loss = self.weight_decay_loss(trainable_variables)
      total_loss = model_loss + l2_regularization_loss
      return {
          'total_loss': total_loss,
          'loss': total_loss,
          'fast_rcnn_class_loss': frcnn_class_loss,
          'fast_rcnn_box_loss': frcnn_box_loss,
          'fast_rcnn_box_score_loss': box_score_loss,
          'mask_loss': mask_loss,
          'model_loss': model_loss,
          'l2_regularization_loss': l2_regularization_loss,
          'rpn_score_loss': rpn_score_loss,
          'rpn_box_loss': rpn_box_loss,
          'rpn_center_loss': rpn_center_loss,
      }

    return _total_loss_fn

  def build_input_layers(self, params, mode):
    is_training = mode == mode_keys.TRAIN
    input_shape = (
        params.olnmask_parser.output_size +
        [params.olnmask_parser.num_channels])
    if is_training:
      batch_size = params.train.batch_size
      input_layer = {
          'image':
              tf.keras.layers.Input(
                  shape=input_shape,
                  batch_size=batch_size,
                  name='image',
                  dtype=tf.bfloat16 if self._use_bfloat16 else tf.float32),
          'image_info':
              tf.keras.layers.Input(
                  shape=[4, 2],
                  batch_size=batch_size,
                  name='image_info',
              ),
          'gt_boxes':
              tf.keras.layers.Input(
                  shape=[params.olnmask_parser.max_num_instances, 4],
                  batch_size=batch_size,
                  name='gt_boxes'),
          'gt_classes':
              tf.keras.layers.Input(
                  shape=[params.olnmask_parser.max_num_instances],
                  batch_size=batch_size,
                  name='gt_classes',
                  dtype=tf.int64),
      }
      if self._include_mask:
        input_layer['gt_masks'] = tf.keras.layers.Input(
            shape=[
                params.olnmask_parser.max_num_instances,
                params.olnmask_parser.mask_crop_size,
                params.olnmask_parser.mask_crop_size
            ],
            batch_size=batch_size,
            name='gt_masks')
    else:
      batch_size = params.eval.batch_size
      input_layer = {
          'image':
              tf.keras.layers.Input(
                  shape=input_shape,
                  batch_size=batch_size,
                  name='image',
                  dtype=tf.bfloat16 if self._use_bfloat16 else tf.float32),
          'image_info':
              tf.keras.layers.Input(
                  shape=[4, 2],
                  batch_size=batch_size,
                  name='image_info',
              ),
      }
    return input_layer

  def build_model(self, params, mode):
    if self._keras_model is None:
      input_layers = self.build_input_layers(self._params, mode)
      with keras_utils.maybe_enter_backend_graph():
        outputs = self.model_outputs(input_layers, mode)

        model = tf.keras.models.Model(
            inputs=input_layers, outputs=outputs, name='olnmask')
        assert model is not None, 'Fail to build tf.keras.Model.'
        model.optimizer = self.build_optimizer()
        self._keras_model = model

    return self._keras_model

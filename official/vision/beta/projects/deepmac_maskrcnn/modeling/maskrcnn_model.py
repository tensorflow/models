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

# Import libraries

from absl import logging
import tensorflow as tf

from official.vision.beta.ops import box_ops


def resize_as(source, size):

  source = tf.transpose(source, (0, 2, 3, 1))
  source = tf.image.resize(source, (size, size))
  return tf.transpose(source, (0, 3, 1, 2))


@tf.keras.utils.register_keras_serializable(package='Vision')
class DeepMaskRCNNModel(tf.keras.Model):
  """The Mask R-CNN model."""

  def __init__(self,
               backbone,
               decoder,
               rpn_head,
               detection_head,
               roi_generator,
               roi_sampler,
               roi_aligner,
               detection_generator,
               mask_head=None,
               mask_sampler=None,
               mask_roi_aligner=None,
               use_gt_boxes_for_masks=False,
               **kwargs):
    """Initializes the Mask R-CNN model.

    Args:
      backbone: `tf.keras.Model`, the backbone network.
      decoder: `tf.keras.Model`, the decoder network.
      rpn_head: the RPN head.
      detection_head: the detection head.
      roi_generator: the ROI generator.
      roi_sampler: the ROI sampler.
      roi_aligner: the ROI aligner.
      detection_generator: the detection generator.
      mask_head: the mask head.
      mask_sampler: the mask sampler.
      mask_roi_aligner: the ROI alginer for mask prediction.
      use_gt_boxes_for_masks: bool, if set, crop using groundtruth boxes
        instead of proposals for training mask head
      **kwargs: keyword arguments to be passed.
    """
    super(DeepMaskRCNNModel, self).__init__(**kwargs)
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
        'use_gt_boxes_for_masks': use_gt_boxes_for_masks
    }
    self.backbone = backbone
    self.decoder = decoder
    self.rpn_head = rpn_head
    self.detection_head = detection_head
    self.roi_generator = roi_generator
    self.roi_sampler = roi_sampler
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

  def call(self,
           images,
           image_shape,
           anchor_boxes=None,
           gt_boxes=None,
           gt_classes=None,
           gt_masks=None,
           training=None):
    model_outputs = {}

    # Feature extraction.
    features = self.backbone(images)
    if self.decoder:
      features = self.decoder(features)

    # Region proposal network.
    rpn_scores, rpn_boxes = self.rpn_head(features)

    model_outputs.update({
        'rpn_boxes': rpn_boxes,
        'rpn_scores': rpn_scores
    })

    # Generate RoIs.
    rois, _ = self.roi_generator(
        rpn_boxes, rpn_scores, anchor_boxes, image_shape, training)

    if training:
      rois = tf.stop_gradient(rois)

      rois, matched_gt_boxes, matched_gt_classes, matched_gt_indices = (
          self.roi_sampler(rois, gt_boxes, gt_classes))
      # Assign target for the 2nd stage classification.
      box_targets = box_ops.encode_boxes(
          matched_gt_boxes, rois, weights=[10.0, 10.0, 5.0, 5.0])
      # If the target is background, the box target is set to all 0s.
      box_targets = tf.where(
          tf.tile(
              tf.expand_dims(tf.equal(matched_gt_classes, 0), axis=-1),
              [1, 1, 4]),
          tf.zeros_like(box_targets),
          box_targets)
      model_outputs.update({
          'class_targets': matched_gt_classes,
          'box_targets': box_targets,
      })

    # RoI align.
    roi_features = self.roi_aligner(features, rois)

    # Detection head.
    raw_scores, raw_boxes = self.detection_head(roi_features)

    if training:
      model_outputs.update({
          'class_outputs': raw_scores,
          'box_outputs': raw_boxes,
      })
    else:
      # Post-processing.
      detections = self.detection_generator(
          raw_boxes, raw_scores, rois, image_shape)
      model_outputs.update({
          'detection_boxes': detections['detection_boxes'],
          'detection_scores': detections['detection_scores'],
          'detection_classes': detections['detection_classes'],
          'num_detections': detections['num_detections'],
      })

    if not self._include_mask:
      return model_outputs

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
            rois,
            matched_gt_boxes,
            matched_gt_classes,
            matched_gt_indices,
            gt_masks)
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
      mask_roi_features = self.mask_roi_aligner(features, gt_boxes)
      raw_masks = self.mask_head([mask_roi_features, gt_classes])

    else:
      mask_roi_features = self.mask_roi_aligner(features, rois)
      raw_masks = self.mask_head([mask_roi_features, roi_classes])

    # Mask head.
    if training:
      model_outputs.update({
          'mask_outputs': raw_masks,
      })
    else:
      model_outputs.update({
          'detection_masks': tf.math.sigmoid(raw_masks),
      })
    return model_outputs

  @property
  def checkpoint_items(self):
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

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)

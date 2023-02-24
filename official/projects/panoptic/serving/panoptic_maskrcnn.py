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

"""Panoptic Segmentation input and model functions for serving/inference."""

from typing import List

import tensorflow as tf

from official.core import config_definitions as cfg
from official.projects.panoptic.modeling import panoptic_maskrcnn_model
from official.vision.serving import detection


class PanopticSegmentationModule(detection.DetectionModule):
  """Panoptic Segmentation Module."""

  def __init__(self,
               params: cfg.ExperimentConfig,
               *,
               model: tf.keras.Model,
               batch_size: int,
               input_image_size: List[int],
               num_channels: int = 3):
    """Initializes panoptic segmentation module for export."""

    if batch_size is None:
      raise ValueError('batch_size cannot be None for panoptic segmentation '
                       'model.')
    if not isinstance(model, panoptic_maskrcnn_model.PanopticMaskRCNNModel):
      raise ValueError('PanopticSegmentationModule module not implemented for '
                       '{} model.'.format(type(model)))

    super().__init__(
        params=params,
        model=model,
        batch_size=batch_size,
        input_image_size=input_image_size,
        num_channels=num_channels)

  def serve(self, images: tf.Tensor):
    """Casts image to float and run inference.

    Args:
      images: uint8 Tensor of shape [batch_size, None, None, 3]
    Returns:
      Tensor holding detection output logits.
    """
    model_params = self.params.task.model
    with tf.device('cpu:0'):
      images = tf.cast(images, dtype=tf.float32)

      # Tensor Specs for map_fn outputs (images, anchor_boxes, and image_info).
      images_spec = tf.TensorSpec(shape=self._input_image_size + [3],
                                  dtype=tf.float32)

      num_anchors = model_params.anchor.num_scales * len(
          model_params.anchor.aspect_ratios) * 4
      anchor_shapes = []
      for level in range(model_params.min_level, model_params.max_level + 1):
        anchor_level_spec = tf.TensorSpec(
            shape=[
                self._input_image_size[0] // 2**level,
                self._input_image_size[1] // 2**level, num_anchors
            ],
            dtype=tf.float32)
        anchor_shapes.append((str(level), anchor_level_spec))

      image_info_spec = tf.TensorSpec(shape=[4, 2], dtype=tf.float32)

      images, anchor_boxes, image_info = tf.nest.map_structure(
          tf.identity,
          tf.map_fn(
              self._build_inputs,
              elems=images,
              fn_output_signature=(images_spec, dict(anchor_shapes),
                                   image_info_spec),
              parallel_iterations=32))

    # To overcome keras.Model extra limitation to save a model with layers that
    # have multiple inputs, we use `model.call` here to trigger the forward
    # path. Note that, this disables some keras magics happens in `__call__`.
    detections = self.model.call(
        images=images,
        image_info=image_info,
        anchor_boxes=anchor_boxes,
        training=False)

    detections.pop('rpn_boxes')
    detections.pop('rpn_scores')
    detections.pop('cls_outputs')
    detections.pop('box_outputs')
    detections.pop('backbone_features')
    detections.pop('decoder_features')
    if model_params.detection_generator.apply_nms:
      # Normalize detection boxes to [0, 1]. Here we first map them to the
      # original image size, then normalize them to [0, 1].
      detections['detection_boxes'] = (
          detections['detection_boxes'] /
          tf.tile(image_info[:, 2:3, :], [1, 1, 2]) /
          tf.tile(image_info[:, 0:1, :], [1, 1, 2]))

      final_outputs = {
          'detection_boxes': detections['detection_boxes'],
          'detection_scores': detections['detection_scores'],
          'detection_classes': detections['detection_classes'],
          'num_detections': detections['num_detections']
      }

      if 'detection_outer_boxes' in detections:
        detections['detection_outer_boxes'] = (
            detections['detection_outer_boxes'] /
            tf.tile(image_info[:, 2:3, :], [1, 1, 2]) /
            tf.tile(image_info[:, 0:1, :], [1, 1, 2]))
        final_outputs['detection_outer_boxes'] = (
            detections['detection_outer_boxes'])
    else:
      final_outputs = {
          'decoded_boxes': detections['decoded_boxes'],
          'decoded_box_scores': detections['decoded_box_scores']
      }
    masks = detections['segmentation_outputs']
    masks = tf.image.resize(masks, self._input_image_size, method='bilinear')
    classes = tf.math.argmax(masks, axis=-1)
    if self.params.task.losses.semantic_segmentation_use_binary_cross_entropy:
      scores = tf.nn.sigmoid(masks)
    else:
      scores = tf.nn.softmax(masks, axis=-1)
    final_outputs.update({
        'detection_masks': detections['detection_masks'],
        'semantic_logits': masks,
        'semantic_scores': scores,
        'semantic_classes': classes,
        'image_info': image_info
    })
    if model_params.generate_panoptic_masks:
      final_outputs.update({
          'panoptic_category_mask':
              detections['panoptic_outputs']['category_mask'],
          'panoptic_instance_mask':
              detections['panoptic_outputs']['instance_mask'],
            })

    return final_outputs

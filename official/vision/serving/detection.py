# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Detection input and model functions for serving/inference."""

import math
from typing import Mapping, Tuple

from absl import logging
import tensorflow as tf, tf_keras

from official.vision import configs
from official.vision.modeling import factory
from official.vision.ops import anchor
from official.vision.ops import box_ops
from official.vision.ops import preprocess_ops
from official.vision.serving import export_base


class DetectionModule(export_base.ExportModule):
  """Detection Module."""

  @property
  def _padded_size(self):
    if self.params.task.train_data.parser.pad:
      return preprocess_ops.compute_padded_size(
          self._input_image_size, 2**self.params.task.model.max_level
      )
    else:
      return self._input_image_size

  def _build_model(self):

    nms_versions_supporting_dynamic_batch_size = {'batched', 'v2', 'v3'}
    nms_version = self.params.task.model.detection_generator.nms_version
    if (self._batch_size is None and
        nms_version not in nms_versions_supporting_dynamic_batch_size):
      logging.info('nms_version is set to `batched` because `%s` '
                   'does not support with dynamic batch size.', nms_version)
      self.params.task.model.detection_generator.nms_version = 'batched'

    input_specs = tf_keras.layers.InputSpec(shape=[
        self._batch_size, *self._padded_size, 3])

    if isinstance(self.params.task.model, configs.maskrcnn.MaskRCNN):
      model = factory.build_maskrcnn(
          input_specs=input_specs, model_config=self.params.task.model)
    elif isinstance(self.params.task.model, configs.retinanet.RetinaNet):
      model = factory.build_retinanet(
          input_specs=input_specs, model_config=self.params.task.model)
    else:
      raise ValueError('Detection module not implemented for {} model.'.format(
          type(self.params.task.model)))

    return model

  def _build_anchor_boxes(self):
    """Builds and returns anchor boxes."""
    model_params = self.params.task.model
    input_anchor = anchor.build_anchor_generator(
        min_level=model_params.min_level,
        max_level=model_params.max_level,
        num_scales=model_params.anchor.num_scales,
        aspect_ratios=model_params.anchor.aspect_ratios,
        anchor_size=model_params.anchor.anchor_size)
    return input_anchor(image_size=self._padded_size)

  def _build_inputs(self, image):
    """Builds detection model inputs for serving."""

    if isinstance(image, tf.RaggedTensor):
      image = image.to_tensor()
    image = tf.cast(image, dtype=tf.float32)

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(
        image, offset=preprocess_ops.MEAN_RGB, scale=preprocess_ops.STDDEV_RGB)

    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._input_image_size,
        padded_size=self._padded_size,
        aug_scale_min=1.0,
        aug_scale_max=1.0,
        keep_aspect_ratio=self.params.task.train_data.parser.keep_aspect_ratio,
    )
    anchor_boxes = self._build_anchor_boxes()

    return image, anchor_boxes, image_info

  def _normalize_coordinates(self, detections_dict, dict_keys, image_info):
    """Normalizes detection coordinates between 0 and 1.

    Args:
      detections_dict: Dictionary containing the output of the model prediction.
      dict_keys: Key names corresponding to the tensors of the output dictionary
        that we want to update.
      image_info: Tensor containing the details of the image resizing.

    Returns:
      detections_dict: Updated detection dictionary.
    """
    for key in dict_keys:
      if key not in detections_dict:
        continue
      detection_boxes = detections_dict[key] / tf.tile(
          image_info[:, 2:3, :], [1, 1, 2]
      )
      detections_dict[key] = box_ops.normalize_boxes(
          detection_boxes, image_info[:, 0:1, :]
      )
      detections_dict[key] = tf.clip_by_value(detections_dict[key], 0.0, 1.0)

    return detections_dict

  def preprocess(
      self, images: tf.Tensor
  ) -> Tuple[tf.Tensor, Mapping[str, tf.Tensor], tf.Tensor]:
    """Preprocesses inputs to be suitable for the model.

    Args:
      images: The images tensor.
    Returns:
      images: The images tensor cast to float.
      anchor_boxes: Dict mapping anchor levels to anchor boxes.
      image_info: Tensor containing the details of the image resizing.

    """
    model_params = self.params.task.model
    with tf.device('cpu:0'):
      # Tensor Specs for map_fn outputs (images, anchor_boxes, and image_info).
      images_spec = tf.TensorSpec(shape=self._padded_size + [3],
                                  dtype=tf.float32)

      num_anchors = model_params.anchor.num_scales * len(
          model_params.anchor.aspect_ratios) * 4
      anchor_shapes = []
      for level in range(model_params.min_level, model_params.max_level + 1):
        anchor_level_spec = tf.TensorSpec(
            shape=[
                math.ceil(self._padded_size[0] / 2**level),
                math.ceil(self._padded_size[1] / 2**level),
                num_anchors,
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

      return images, anchor_boxes, image_info

  def serve(self, images: tf.Tensor):
    """Casts image to float and runs inference.

    Args:
      images: uint8 Tensor of shape [batch_size, None, None, 3]
    Returns:
      Tensor holding detection output logits.
    """

    # Skip image preprocessing when input_type is tflite so it is compatible
    # with TFLite quantization.
    if self._input_type != 'tflite':
      images, anchor_boxes, image_info = self.preprocess(images)
    else:
      with tf.device('cpu:0'):
        anchor_boxes = self._build_anchor_boxes()
        # image_info is a 3D tensor of shape [batch_size, 4, 2]. It is in the
        # format of [[original_height, original_width],
        # [desired_height, desired_width], [y_scale, x_scale],
        # [y_offset, x_offset]]. When input_type is tflite, input image is
        # supposed to be preprocessed already.
        image_info = tf.convert_to_tensor([[
            self._input_image_size, self._input_image_size, [1.0, 1.0], [0, 0]
        ]],
                                          dtype=tf.float32)
    input_image_shape = image_info[:, 1, :]

    # To overcome keras.Model extra limitation to save a model with layers that
    # have multiple inputs, we use `model.call` here to trigger the forward
    # path. Note that, this disables some keras magics happens in `__call__`.
    model_call_kwargs = {
        'images': images,
        'image_shape': input_image_shape,
        'anchor_boxes': anchor_boxes,
        'training': False,
    }
    if isinstance(self.params.task.model, configs.retinanet.RetinaNet):
      model_call_kwargs['output_intermediate_features'] = (
          self.params.task.export_config.output_intermediate_features
      )
    detections = self.model.call(**model_call_kwargs)

    if self.params.task.model.detection_generator.apply_nms:
      # For RetinaNet model, apply export_config.
      # TODO(huizhongc): Add export_config to fasterrcnn and maskrcnn as needed.
      if isinstance(self.params.task.model, configs.retinanet.RetinaNet):
        export_config = self.params.task.export_config
        # Normalize detection box coordinates to [0, 1].
        if export_config.output_normalized_coordinates:
          keys = ['detection_boxes', 'detection_outer_boxes']
          detections = self._normalize_coordinates(detections, keys, image_info)

        # Cast num_detections and detection_classes to float. This allows the
        # model inference to work on chain (go/chain) as chain requires floating
        # point outputs.
        if export_config.cast_num_detections_to_float:
          detections['num_detections'] = tf.cast(
              detections['num_detections'], dtype=tf.float32)
        if export_config.cast_detection_classes_to_float:
          detections['detection_classes'] = tf.cast(
              detections['detection_classes'], dtype=tf.float32)

      final_outputs = {
          'detection_boxes': detections['detection_boxes'],
          'detection_scores': detections['detection_scores'],
          'detection_classes': detections['detection_classes'],
          'num_detections': detections['num_detections']
      }
      if 'detection_outer_boxes' in detections:
        final_outputs['detection_outer_boxes'] = (
            detections['detection_outer_boxes'])
    else:
      # For RetinaNet model, apply export_config.
      if isinstance(self.params.task.model, configs.retinanet.RetinaNet):
        export_config = self.params.task.export_config
        # Normalize detection box coordinates to [0, 1].
        if export_config.output_normalized_coordinates:
          keys = ['decoded_boxes']
          detections = self._normalize_coordinates(detections, keys, image_info)
      final_outputs = {
          'decoded_boxes': detections['decoded_boxes'],
          'decoded_box_scores': detections['decoded_box_scores']
      }

    if 'detection_masks' in detections.keys():
      final_outputs['detection_masks'] = detections['detection_masks']
    if (
        isinstance(self.params.task.model, configs.retinanet.RetinaNet)
        and self.params.task.export_config.output_intermediate_features
    ):
      final_outputs.update(
          {
              k: v
              for k, v in detections.items()
              if k.startswith('backbone_') or k.startswith('decoder_')
          }
      )

    if self.params.task.model.detection_generator.nms_version != 'tflite':
      final_outputs.update({'image_info': image_info})
    return final_outputs

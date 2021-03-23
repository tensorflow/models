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

# Lint as: python3
"""Detection input and model functions for serving/inference."""

import tensorflow as tf

from official.vision.beta import configs
from official.vision.beta.modeling import factory
from official.vision.beta.ops import anchor
from official.vision.beta.ops import preprocess_ops
from official.vision.beta.serving import export_base


MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


class DetectionModule(export_base.ExportModule):
  """Detection Module."""

  def _build_model(self):

    if self._batch_size is None:
      ValueError("batch_size can't be None for detection models")
    if not self.params.task.model.detection_generator.use_batched_nms:
      ValueError('Only batched_nms is supported.')
    input_specs = tf.keras.layers.InputSpec(shape=[self._batch_size] +
                                            self._input_image_size + [3])

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

  def _build_inputs(self, image):
    """Builds detection model inputs for serving."""
    model_params = self.params.task.model
    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image,
                                           offset=MEAN_RGB,
                                           scale=STDDEV_RGB)

    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._input_image_size,
        padded_size=preprocess_ops.compute_padded_size(
            self._input_image_size, 2**model_params.max_level),
        aug_scale_min=1.0,
        aug_scale_max=1.0)

    input_anchor = anchor.build_anchor_generator(
        min_level=model_params.min_level,
        max_level=model_params.max_level,
        num_scales=model_params.anchor.num_scales,
        aspect_ratios=model_params.anchor.aspect_ratios,
        anchor_size=model_params.anchor.anchor_size)
    anchor_boxes = input_anchor(image_size=(self._input_image_size[0],
                                            self._input_image_size[1]))

    return image, anchor_boxes, image_info

  def serve(self, images: tf.Tensor):
    """Cast image to float and run inference.

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

    input_image_shape = image_info[:, 1, :]

    detections = self.model.call(
        images=images,
        image_shape=input_image_shape,
        anchor_boxes=anchor_boxes,
        training=False)

    final_outputs = {
        'detection_boxes': detections['detection_boxes'],
        'detection_scores': detections['detection_scores'],
        'detection_classes': detections['detection_classes'],
        'num_detections': detections['num_detections'],
        'image_info': image_info
    }
    if 'detection_masks' in detections.keys():
      final_outputs['detection_masks'] = detections['detection_masks']

    return final_outputs

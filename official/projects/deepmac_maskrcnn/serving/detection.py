# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

from typing import Dict, Mapping, Text

import tensorflow as tf, tf_keras

from official.projects.deepmac_maskrcnn.configs import deep_mask_head_rcnn as cfg
from official.projects.deepmac_maskrcnn.modeling import maskrcnn_model
from official.projects.deepmac_maskrcnn.tasks import deep_mask_head_rcnn
from official.vision.ops import box_ops
from official.vision.serving import detection


def reverse_input_box_transformation(boxes, image_info):
  """Reverse the Mask R-CNN model's input boxes tranformation.

  Args:
    boxes: A [batch_size, num_boxes, 4] float tensor of boxes in normalized
      coordinates.
    image_info: a 2D `Tensor` that encodes the information of the image and the
      applied preprocessing. It is in the format of
      [[original_height, original_width], [desired_height, desired_width],
       [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
      desired_width] is the actual scaled image size, and [y_scale, x_scale] is
      the scaling factor, which is the ratio of
      scaled dimension / original dimension.

  Returns:
    boxes: Same shape as input `boxes` but in the absolute coordinate space of
      the preprocessed image.
  """
  # Reversing sequence from Detection_module.serve when
  # output_normalized_coordinates=true
  scale = image_info[:, 2:3, :]
  scale = tf.tile(scale, [1, 1, 2])
  boxes = boxes * scale
  height_width = image_info[:, 0:1, :]
  return box_ops.denormalize_boxes(boxes, height_width)


class DetectionModule(detection.DetectionModule):
  """Detection Module."""

  def _build_model(self):

    if self._batch_size is None:
      ValueError("batch_size can't be None for detection models")
    if self.params.task.model.detection_generator.nms_version != 'batched':
      ValueError('Only batched_nms is supported.')
    input_specs = tf_keras.layers.InputSpec(shape=[self._batch_size] +
                                            self._input_image_size + [3])

    if isinstance(self.params.task.model, cfg.DeepMaskHeadRCNN):
      model = deep_mask_head_rcnn.build_maskrcnn(
          input_specs=input_specs, model_config=self.params.task.model)
    else:
      raise ValueError('Detection module not implemented for {} model.'.format(
          type(self.params.task.model)))

    return model

  @tf.function
  def inference_for_tflite_image_and_boxes(
      self, images: tf.Tensor, boxes: tf.Tensor) -> Mapping[str, tf.Tensor]:
    """A tf-function for serve_image_and_boxes.

    Args:
      images: A [batch_size, height, width, channels] float tensor.
      boxes: A [batch_size, num_boxes, 4] float tensor containing boxes
        normalized to the input image.

    Returns:
      result: A dict containing:
        'detection_masks': A [batch_size, num_boxes, mask_height, mask_width]
          float tensor containing per-pixel mask probabilities.
    """

    if not isinstance(self.model, maskrcnn_model.DeepMaskRCNNModel):
      raise ValueError(
          ('Can only use image and boxes input for DeepMaskRCNNModel, '
           'Found {}'.format(type(self.model))))

    return self.serve_image_and_boxes(images, boxes)

  def serve_image_and_boxes(self, images: tf.Tensor, boxes: tf.Tensor):
    """Function used to export a model that consumes and image and boxes.

    The model predicts the class-agnostic masks at the given box locations.

    Args:
      images: A [batch_size, height, width, channels] float tensor.
      boxes: A [batch_size, num_boxes, 4] float tensor containing boxes
        normalized to the input image.

    Returns:
      result: A dict containing:
        'detection_masks': A [batch_size, num_boxes, mask_height, mask_width]
          float tensor containing per-pixel mask probabilities.
    """
    images, _, image_info = self.preprocess(images)
    boxes = reverse_input_box_transformation(boxes, image_info)
    result = self.model.call_images_and_boxes(images, boxes)
    return result

  def get_inference_signatures(self, function_keys: Dict[Text, Text]):
    signatures = {}

    if 'image_and_boxes_tensor' in function_keys:
      def_name = function_keys['image_and_boxes_tensor']
      image_signature = tf.TensorSpec(
          shape=[self._batch_size] + [None] * len(self._input_image_size) +
          [self._num_channels],
          dtype=tf.uint8)
      boxes_signature = tf.TensorSpec(shape=[self._batch_size, None, 4],
                                      dtype=tf.float32)
      tf_function = self.inference_for_tflite_image_and_boxes
      signatures[def_name] = tf_function.get_concrete_function(
          image_signature, boxes_signature)

    function_keys.pop('image_and_boxes_tensor', None)
    parent_signatures = super(DetectionModule, self).get_inference_signatures(
        function_keys)
    signatures.update(parent_signatures)

    return signatures

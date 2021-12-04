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
"""Semantic segmentation input and model functions for serving/inference."""

import tensorflow as tf

from official.vision.beta.modeling import factory
from official.vision.beta.ops import preprocess_ops
from official.vision.beta.serving import export_base


MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


class SegmentationModule(export_base.ExportModule):
  """Segmentation Module."""

  def _build_model(self):
    input_specs = tf.keras.layers.InputSpec(
        shape=[self._batch_size] + self._input_image_size + [3])

    return factory.build_segmentation_model(
        input_specs=input_specs,
        model_config=self.params.task.model,
        l2_regularizer=None)

  def _build_inputs(self, image):
    """Builds classification model inputs for serving."""

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image,
                                           offset=MEAN_RGB,
                                           scale=STDDEV_RGB)

    image, _ = preprocess_ops.resize_and_crop_image(
        image,
        self._input_image_size,
        padded_size=self._input_image_size,
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    return image

  def serve(self, images):
    """Cast image to float and run inference.

    Args:
      images: uint8 Tensor of shape [batch_size, None, None, 3]
    Returns:
      Tensor holding classification output logits.
    """
    # Skip image preprocessing when input_type is tflite so it is compatible
    # with TFLite quantization.
    if self._input_type != 'tflite':
      with tf.device('cpu:0'):
        images = tf.cast(images, dtype=tf.float32)

        images = tf.nest.map_structure(
            tf.identity,
            tf.map_fn(
                self._build_inputs,
                elems=images,
                fn_output_signature=tf.TensorSpec(
                    shape=self._input_image_size + [3], dtype=tf.float32),
                parallel_iterations=32))

    masks = self.inference_step(images)
    masks = tf.image.resize(masks, self._input_image_size, method='bilinear')

    return dict(predicted_masks=masks)

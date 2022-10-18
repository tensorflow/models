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

"""Semantic segmentation input and model functions for serving/inference."""

import tensorflow as tf

from official.vision.modeling import factory
from official.vision.ops import preprocess_ops
from official.vision.serving import export_base


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
    image = preprocess_ops.normalize_image(
        image, offset=preprocess_ops.MEAN_RGB, scale=preprocess_ops.STDDEV_RGB)

    if self.params.task.train_data.preserve_aspect_ratio:
      image, image_info = preprocess_ops.resize_and_crop_image(
          image,
          self._input_image_size,
          padded_size=self._input_image_size,
          aug_scale_min=1.0,
          aug_scale_max=1.0)
    else:
      image, image_info = preprocess_ops.resize_image(image,
                                                      self._input_image_size)
    return image, image_info

  def serve(self, images):
    """Cast image to float and run inference.

    Args:
      images: uint8 Tensor of shape [batch_size, None, None, 3]
    Returns:
      Tensor holding classification output logits.
    """
    # Skip image preprocessing when input_type is tflite so it is compatible
    # with TFLite quantization.
    image_info = None
    if self._input_type != 'tflite':
      with tf.device('cpu:0'):
        images = tf.cast(images, dtype=tf.float32)
        images_spec = tf.TensorSpec(
            shape=self._input_image_size + [3], dtype=tf.float32)
        image_info_spec = tf.TensorSpec(shape=[4, 2], dtype=tf.float32)

        images, image_info = tf.nest.map_structure(
            tf.identity,
            tf.map_fn(
                self._build_inputs,
                elems=images,
                fn_output_signature=(images_spec, image_info_spec),
                parallel_iterations=32))

    outputs = self.inference_step(images)

    # Optionally resize prediction to the input image size.
    if self.params.task.export_config.rescale_output:
      logits = outputs['logits']
      if logits.shape[0] != 1:
        raise ValueError('Batch size cannot be more than 1.')

      image_shape = tf.cast(image_info[0, 0, :], tf.int32)
      if self.params.task.train_data.preserve_aspect_ratio:
        rescale_size = tf.cast(
            tf.math.ceil(image_info[0, 1, :] / image_info[0, 2, :]), tf.int32)
        offsets = tf.cast(image_info[0, 3, :], tf.int32)
        logits = tf.image.resize(logits, rescale_size, method='bilinear')
        outputs['logits'] = tf.image.crop_to_bounding_box(
            logits, offsets[0], offsets[1], image_shape[0], image_shape[1])
      else:
        outputs['logits'] = tf.image.resize(
            logits, [image_shape[0], image_shape[1]], method='bilinear')
    else:
      outputs['logits'] = tf.image.resize(
          outputs['logits'], self._input_image_size, method='bilinear')

    if image_info is not None:
      outputs.update({'image_info': image_info})

    return outputs

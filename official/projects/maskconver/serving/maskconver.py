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

"""Maskconver input and model functions for serving/inference."""

import tensorflow as tf, tf_keras

from official.projects.maskconver.modeling import factory
from official.vision.ops import preprocess_ops
from official.vision.serving import export_base


class MaskConverModule(export_base.ExportModule):
  """MaskConver Module."""

  def _build_model(self):
    input_specs = tf_keras.layers.InputSpec(
        shape=[self._batch_size] + self._input_image_size + [3])

    return factory.build_maskconver_model(
        input_specs=input_specs,
        model_config=self.params.task.model,
        l2_regularizer=None)

  def _build_inputs(self, image):
    """Builds MaskConver model inputs for serving."""

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(
        image, offset=preprocess_ops.MEAN_RGB, scale=preprocess_ops.STDDEV_RGB)

    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._input_image_size,
        padded_size=self._input_image_size,
        aug_scale_min=1.0,
        aug_scale_max=1.0)
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
    if self._input_type != "tflite":
      with tf.device("cpu:0"):
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

    outputs = self.inference_step(images, image_info)

    if "panoptic_outputs" in outputs:
      outputs.update({
          "panoptic_category_mask":
              outputs["panoptic_outputs"]["category_mask"],
          "panoptic_instance_mask":
              outputs["panoptic_outputs"]["instance_mask"],
      })
      del outputs["panoptic_outputs"]

    if image_info is not None:
      outputs.update({"image_info": image_info})

    return outputs

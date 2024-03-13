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

"""Image classification input and model functions for serving/inference."""

import tensorflow as tf, tf_keras

from official.vision.modeling import factory
from official.vision.ops import preprocess_ops
from official.vision.serving import export_base


class ClassificationModule(export_base.ExportModule):
  """classification Module."""

  def _build_model(self):
    input_specs = tf_keras.layers.InputSpec(
        shape=[self._batch_size] + self._input_image_size + [3])

    return factory.build_classification_model(
        input_specs=input_specs,
        model_config=self.params.task.model,
        l2_regularizer=None)

  def _crop_and_resize(self, image):
    if self.params.task.train_data.aug_crop:
      image = preprocess_ops.center_crop_image(image)

    image = tf.image.resize(
        image, self._input_image_size, method=tf.image.ResizeMethod.BILINEAR)

    image = tf.reshape(
        image, [self._input_image_size[0], self._input_image_size[1], 3])

    return image

  def _build_inputs(self, image):
    """Builds classification model inputs for serving."""
    # Center crops and resizes image.
    if isinstance(image, tf.RaggedTensor):
      image = image.to_tensor()
    image = tf.cast(image, dtype=tf.float32)

    # For these input types, decode_image already performs cropping.
    if not (
        self._input_type in ['tf_example', 'image_bytes']
        and len(self._input_image_size) == 2):
      image = self._crop_and_resize(image)

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(
        image, offset=preprocess_ops.MEAN_RGB, scale=preprocess_ops.STDDEV_RGB)
    return image

  def _decode_image(self, encoded_image_bytes: str) -> tf.Tensor:
    """Decodes an image bytes to an image tensor.

    Use `tf.image.decode_image` to decode an image if input is expected to be 2D
    image; otherwise use `tf.io.decode_raw` to convert the raw bytes to tensor
    and reshape it to desire shape.

    Args:
      encoded_image_bytes: An encoded image string to be decoded.

    Returns:
      A decoded image tensor.
    """
    if len(self._input_image_size) == 2:
      # Decode an image if 2D input is expected.
      image_tensor = tf.image.decode_image(
          encoded_image_bytes, channels=self._num_channels
      )
      image_tensor.set_shape((None, None, self._num_channels))
      # Crop the image inside the same loop as decoding an image
      # if there could be several images of different sizes in the batch.
      image_tensor = tf.cast(image_tensor, dtype=tf.float32)
      image_tensor = self._crop_and_resize(image_tensor)
      image_tensor = tf.cast(image_tensor, tf.uint8)
      return image_tensor
    else:
      # Convert raw bytes into a tensor and reshape it, if not 2D input.
      image_tensor = tf.io.decode_raw(encoded_image_bytes, out_type=tf.uint8)
      image_tensor = tf.reshape(
          image_tensor, self._input_image_size + [self._num_channels]
      )
    return image_tensor

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
        images = tf.nest.map_structure(
            tf.identity,
            tf.map_fn(
                self._build_inputs,
                elems=images,
                fn_output_signature=tf.TensorSpec(
                    shape=self._input_image_size + [3], dtype=tf.float32),
                parallel_iterations=32))

    logits = self.inference_step(images)
    if self.params.task.train_data.is_multilabel:
      probs = tf.math.sigmoid(logits)
    else:
      probs = tf.nn.softmax(logits)

    return {'logits': logits, 'probs': probs}

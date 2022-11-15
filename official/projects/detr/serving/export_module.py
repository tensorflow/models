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

"""Export module for DETR model."""
import tensorflow as tf

from official.projects.detr.modeling import detr
from official.vision.modeling import backbones
from official.vision.ops import preprocess_ops
from official.vision.serving import detection


class DETRModule(detection.DetectionModule):
  """DETR detection module."""

  def _build_model(self) -> tf.keras.Model:
    input_specs = tf.keras.layers.InputSpec(shape=[self._batch_size] +
                                            self._input_image_size +
                                            [self._num_channels])

    backbone = backbones.factory.build_backbone(
        input_specs=input_specs,
        backbone_config=self.params.task.model.backbone,
        norm_activation_config=self.params.task.model.norm_activation)

    model = detr.DETR(backbone, self.params.task.model.backbone_endpoint_name,
                      self.params.task.model.num_queries,
                      self.params.task.model.hidden_size,
                      self.params.task.model.num_classes,
                      self.params.task.model.num_encoder_layers,
                      self.params.task.model.num_decoder_layers)
    model(tf.keras.Input(input_specs.shape[1:]))
    return model

  def _build_inputs(self, image: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Builds detection model inputs for serving."""
    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(
        image, offset=preprocess_ops.MEAN_RGB, scale=preprocess_ops.STDDEV_RGB)

    image, image_info = preprocess_ops.resize_image(
        image, size=self._input_image_size)

    return image, image_info

  def serve(self, images: tf.Tensor) -> dict[str, tf.Tensor]:
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

    outputs = self.inference_step(images)[-1]
    outputs = {
        'detection_boxes': outputs['detection_boxes'],
        'detection_scores': outputs['detection_scores'],
        'detection_classes': outputs['detection_classes'],
        'num_detections': outputs['num_detections']
    }
    if image_info is not None:
      outputs['detection_boxes'] = outputs['detection_boxes'] * tf.expand_dims(
          tf.concat([
              image_info[:, 1:2, 0], image_info[:, 1:2, 1],
              image_info[:, 1:2, 0], image_info[:, 1:2, 1]
          ],
                    axis=1),
          axis=1)

      outputs.update({'image_info': image_info})

    return outputs

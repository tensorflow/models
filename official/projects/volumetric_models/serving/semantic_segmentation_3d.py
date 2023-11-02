# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""3D semantic segmentation input and model functions for serving/inference."""

from typing import Mapping

import tensorflow as tf, tf_keras

# pylint: disable=unused-import
from official.projects.volumetric_models.modeling import backbones
from official.projects.volumetric_models.modeling import decoders
from official.projects.volumetric_models.modeling import factory
from official.vision.serving import export_base


class SegmentationModule(export_base.ExportModule):
  """Segmentation Module."""

  def _build_model(self) -> tf_keras.Model:
    """Builds and returns a segmentation model."""
    num_channels = self.params.task.model.num_channels
    input_specs = tf_keras.layers.InputSpec(
        shape=[self._batch_size] + self._input_image_size + [num_channels])

    return factory.build_segmentation_model_3d(
        input_specs=input_specs,
        model_config=self.params.task.model,
        l2_regularizer=None)

  def serve(
      self, images: tf.Tensor) -> Mapping[str, tf.Tensor]:
    """Casts an image tensor to float and runs inference.

    Args:
      images: A uint8 tf.Tensor of shape [batch_size, None, None, None,
        num_channels].

    Returns:
      A dictionary holding segmentation outputs.
    """
    with tf.device('cpu:0'):
      images = tf.cast(images, dtype=tf.float32)

    outputs = self.inference_step(images)
    output_key = 'logits' if self.params.task.model.head.output_logits else 'probs'

    return {output_key: outputs['logits']}

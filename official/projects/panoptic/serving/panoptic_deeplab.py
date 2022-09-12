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
from official.projects.panoptic.modeling import factory
from official.projects.panoptic.modeling import panoptic_deeplab_model
from official.vision.serving import semantic_segmentation


class PanopticSegmentationModule(
    semantic_segmentation.SegmentationModule):
  """Panoptic Deeplab Segmentation Module."""

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
    if not isinstance(model, panoptic_deeplab_model.PanopticDeeplabModel):
      raise ValueError('PanopticSegmentationModule module not '
                       'implemented for {} model.'.format(type(model)))
    params.task.train_data.preserve_aspect_ratio = True
    super(PanopticSegmentationModule, self).__init__(
        params=params,
        model=model,
        batch_size=batch_size,
        input_image_size=input_image_size,
        num_channels=num_channels)

  def _build_model(self):
    input_specs = tf.keras.layers.InputSpec(shape=[self._batch_size] +
                                            self._input_image_size + [3])

    return factory.build_panoptic_deeplab(
        input_specs=input_specs,
        model_config=self.params.task.model,
        l2_regularizer=None)

  def serve(self, images: tf.Tensor):
    """Cast image to float and run inference.

    Args:
      images: uint8 Tensor of shape [batch_size, None, None, 3]

    Returns:
      Tensor holding detection output logits.
    """
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

    outputs = self.model.call(
        inputs=images, image_info=image_info, training=False)

    masks = outputs['segmentation_outputs']
    masks = tf.image.resize(masks, self._input_image_size, method='bilinear')
    classes = tf.math.argmax(masks, axis=-1)
    scores = tf.nn.softmax(masks, axis=-1)
    final_outputs = {
        'semantic_logits': masks,
        'semantic_scores': scores,
        'semantic_classes': classes,
        'image_info': image_info,
        'panoptic_category_mask': outputs['category_mask'],
        'panoptic_instance_mask': outputs['instance_mask'],
    }

    return final_outputs
